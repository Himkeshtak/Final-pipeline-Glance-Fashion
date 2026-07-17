"""Command-line entry point for a complete fashion retrieval cycle."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from fashion_indexer.config import COLLECTION_NAME, ModelSettings
from fashion_indexer.device import resolve_device
from fashion_indexer.environment import load_environment
from fashion_indexer.errors import FashionIndexerError
from fashion_retriever.config import (
    DEFAULT_OPENROUTER_MODEL,
    DEFAULT_QUERY,
    LLMSettings,
    RetrieverSettings,
)
from fashion_retriever.database import FashionSearchStore
from fashion_retriever.encoders import MultiExpertTextEncoder
from fashion_retriever.errors import RetrieverError
from fashion_retriever.llm import OpenRouterQueryParser
from fashion_retriever.pipeline import FashionRetriever

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retrieve fashion images by routing text across three Qdrant named vectors."
    )
    parser.add_argument("query", nargs="?", default=DEFAULT_QUERY)
    parser.add_argument("--db-path", type=Path, default=Path("qdrant_fashion_db"))
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--candidate-limit", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--missing-score", type=float, default=0.0)
    parser.add_argument("--query-attempts", type=int, default=3)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-timeout", type=float, default=45.0)
    parser.add_argument("--llm-transport-retries", type=int, default=2)
    parser.add_argument("--llm-validation-attempts", type=int, default=2)
    parser.add_argument("--model-cache-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument(
        "--precision",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument(
        "--sequential-text-encoding",
        action="store_true",
        help="Run the three independent text towers sequentially instead of concurrently.",
    )
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--json-indent", type=int, default=2)
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
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise RetrieverError(
                "OPENROUTER_API_KEY is not configured in the process environment or .env file."
            )

        llm_settings = LLMSettings(
            model=args.llm_model
            or os.environ.get("OPENROUTER_MODEL", "").strip()
            or DEFAULT_OPENROUTER_MODEL,
            timeout_seconds=args.llm_timeout,
            transport_retries=args.llm_transport_retries,
            validation_attempts=args.llm_validation_attempts,
        )
        retrieval_settings = RetrieverSettings(
            db_path=args.db_path,
            collection_name=args.collection,
            candidate_limit=args.candidate_limit,
            top_k=args.top_k,
            missing_score=args.missing_score,
            query_attempts=args.query_attempts,
        )
        model_settings = ModelSettings(cache_dir=args.model_cache_dir)

        if environment.dotenv_path is not None:
            LOGGER.info("Loaded environment variables from %s", environment.dotenv_path)
        device_spec = resolve_device(args.device, args.precision)
        LOGGER.info("Selected compute device: %s", device_spec.description)

        with FashionSearchStore(
            retrieval_settings.db_path,
            retrieval_settings.collection_name,
            query_attempts=retrieval_settings.query_attempts,
        ) as store:
            count = store.validate_ready(model_settings)
            LOGGER.info("Validated %d indexed fashion image(s)", count)
            query_parser = OpenRouterQueryParser(api_key, llm_settings)
            encoder = MultiExpertTextEncoder(
                model_settings,
                device_spec,
                parallel=not args.sequential_text_encoding,
            )
            retriever = FashionRetriever(query_parser, encoder, store, retrieval_settings)
            response = retriever.search(args.query)

        print(response.model_dump_json(indent=args.json_indent))
        return 0
    except (RetrieverError, FashionIndexerError) as exc:
        LOGGER.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        LOGGER.warning("Retrieval interrupted by user.")
        return 130
    except Exception as exc:
        LOGGER.exception("Unexpected fatal retrieval error: %s", exc)
        return 2


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stderr,
    )
