"""Command-line orchestration for the complete indexing lifecycle."""

from __future__ import annotations

import argparse
import logging
import random
import sys
from collections.abc import Sequence
from pathlib import Path

from fashion_indexer.config import (
    COLLECTION_NAME,
    ModelSettings,
    PipelineSettings,
    build_index_manifest,
)
from fashion_indexer.database import FashionVectorStore
from fashion_indexer.device import resolve_device
from fashion_indexer.environment import load_environment
from fashion_indexer.errors import FashionIndexerError
from fashion_indexer.experts import MultiExpertEncoder
from fashion_indexer.pipeline import FashionIndexingPipeline
from fashion_indexer.preprocessing import ImagePreprocessor

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Index fashion images into a local Qdrant collection with garment, region, "
            "and vibe named vectors."
        )
    )
    parser.add_argument("--images-dir", type=Path, default=Path("images"))
    parser.add_argument("--db-path", type=Path, default=Path("qdrant_fashion_db"))
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Randomly select at most this many discovered images before indexing.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=2020,
        help="Deterministic seed used with --max-images (default: 2020).",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recursively scan the images directory (default: true).",
    )
    parser.add_argument("--max-image-pixels", type=int, default=100_000_000)
    parser.add_argument("--max-image-dimension", type=int, default=2_048)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument(
        "--precision",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--model-cache-dir", type=Path, default=None)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional dotenv file; otherwise .env is discovered from the working directory.",
    )
    parser.add_argument(
        "--recreate-collection",
        action="store_true",
        help="Delete and rebuild the configured collection before indexing.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Do not compare input MD5 hashes with payloads from previous runs.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.log_level)

    try:
        environment = load_environment(args.env_file)
        if environment.dotenv_path is not None:
            LOGGER.info("Loaded environment variables from %s", environment.dotenv_path)
        if environment.huggingface_token_configured:
            LOGGER.info(
                "Hugging Face authentication is configured via %s",
                environment.huggingface_token_source,
            )
        else:
            LOGGER.info("No Hugging Face token is configured; public model access will be used")

        pipeline_settings = PipelineSettings(
            images_dir=args.images_dir,
            db_path=args.db_path,
            collection_name=args.collection,
            recursive=args.recursive,
            batch_size=args.batch_size,
            max_image_pixels=args.max_image_pixels,
            max_image_dimension=args.max_image_dimension,
            device=args.device,
            precision=args.precision,
            recreate_collection=args.recreate_collection,
            skip_existing=not args.no_skip_existing,
        )
        model_settings = ModelSettings(
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            cache_dir=args.model_cache_dir,
        )
        preprocessor = ImagePreprocessor(
            pipeline_settings.images_dir,
            recursive=pipeline_settings.recursive,
            max_image_pixels=pipeline_settings.max_image_pixels,
            max_image_dimension=pipeline_settings.max_image_dimension,
            extensions=pipeline_settings.image_extensions,
        )
        discovered_paths = preprocessor.discover()
        paths = select_random_subset(
            discovered_paths,
            max_images=args.max_images,
            seed=args.shuffle_seed,
        )
        device_spec = resolve_device(pipeline_settings.device, pipeline_settings.precision)
        LOGGER.info("Selected compute device: %s", device_spec.description)
        LOGGER.info("Discovered %d supported image file(s)", len(discovered_paths))
        if len(paths) != len(discovered_paths):
            LOGGER.info(
                "Selected a reproducible random subset of %d image(s) with seed %d",
                len(paths),
                args.shuffle_seed,
            )
        index_manifest = build_index_manifest(
            model_settings,
            pipeline_settings,
            compute_dtype=str(device_spec.dtype).removeprefix("torch."),
        )

        # For an explicitly destructive rebuild, prove every model can initialize before deletion.
        encoder = (
            MultiExpertEncoder(model_settings, device_spec)
            if paths and pipeline_settings.recreate_collection
            else None
        )

        with FashionVectorStore(
            pipeline_settings.db_path,
            pipeline_settings.collection_name,
            index_manifest=index_manifest,
        ) as store:
            store.ensure_collection(recreate=pipeline_settings.recreate_collection)
            existing_hashes = (
                store.existing_file_hashes() if pipeline_settings.skip_existing else set()
            )
            LOGGER.info("Loaded %d existing image hash(es)", len(existing_hashes))

            if not paths:
                count = store.count()
                LOGGER.warning("No supported images found; model loading was skipped.")
                print(
                    f"Verification: collection={pipeline_settings.collection_name!r}, "
                    f"points={count}"
                )
                return 0

            if encoder is None:
                encoder = MultiExpertEncoder(model_settings, device_spec)
            pipeline = FashionIndexingPipeline(
                preprocessor,
                encoder,
                store,
                batch_size=pipeline_settings.batch_size,
                existing_hashes=existing_hashes,
            )
            stats = pipeline.run(paths)
            count = store.count()

        print(
            "Indexing complete: "
            f"discovered={stats.discovered}, indexed={stats.indexed}, "
            f"duplicates={stats.duplicate_files}, already_indexed={stats.already_indexed}, "
            f"invalid={stats.invalid_images}, failures={stats.inference_failures}"
        )
        print(f"Verification: collection={pipeline_settings.collection_name!r}, points={count}")
        return 1 if stats.inference_failures else 0
    except FashionIndexerError as exc:
        LOGGER.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        LOGGER.warning("Indexing interrupted by user.")
        return 130
    except Exception as exc:
        LOGGER.exception("Unexpected fatal error: %s", exc)
        return 2


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stderr,
    )


def select_random_subset(
    paths: Sequence[Path],
    *,
    max_images: int | None,
    seed: int,
) -> list[Path]:
    """Return all paths or a reproducibly shuffled subset without replacement."""

    selected = list(paths)
    if max_images is None:
        return selected
    if max_images < 1:
        raise FashionIndexerError("--max-images must be at least 1.")
    random.Random(seed).shuffle(selected)
    return selected[:max_images]
