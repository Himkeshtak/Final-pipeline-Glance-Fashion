"""Configuration objects and fixed Strategy A vector dimensions."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from fashion_indexer.errors import ConfigurationError

COLLECTION_NAME = "fashion_multi_expert_search"

GARMENT_VECTOR_NAME = "garment_vector"
REGION_VECTOR_NAME = "region_vector"
VIBE_VECTOR_NAME = "vibe_vector"

GARMENT_VECTOR_SIZE = 512
REGION_VECTOR_SIZE = 768
VIBE_VECTOR_SIZE = 768

DEFAULT_GARMENT_LABELS = ("garment", "clothing", "apparel", "dress")
DEFAULT_IMAGE_EXTENSIONS = frozenset(
    {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)


@dataclass(frozen=True, slots=True)
class ModelSettings:
    """Model identifiers and garment-detection parameters."""

    detector_model_id: str = "IDEA-Research/grounding-dino-tiny"
    detector_revision: str = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
    fashion_model_id: str = "patrickjohncyh/fashion-clip"
    fashion_revision: str = "7e3ba62ce16b379a1ab479346b66f192e76f51b7"
    region_model_id: str = "openai/clip-vit-large-patch14"
    region_revision: str = "32bd64288804d66eefd0ccbe215aa642df71cc41"
    vibe_model_id: str = "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"
    vibe_revision: str = "84c9828e63dc9a9351d1fe637c346d4c1c4db341"
    garment_labels: tuple[str, ...] = DEFAULT_GARMENT_LABELS
    box_threshold: float = 0.25
    text_threshold: float = 0.25
    cache_dir: Path | None = None

    def __post_init__(self) -> None:
        if not self.garment_labels:
            raise ConfigurationError("At least one garment detector label is required.")
        if any(not label.strip() for label in self.garment_labels):
            raise ConfigurationError("Garment detector labels must be non-empty strings.")
        revisions = (
            self.detector_revision,
            self.fashion_revision,
            self.region_revision,
            self.vibe_revision,
        )
        if any(
            len(revision) != 40
            or any(character not in "0123456789abcdef" for character in revision.lower())
            for revision in revisions
        ):
            raise ConfigurationError(
                "Model revisions must be full 40-character hexadecimal hashes."
            )
        for name, threshold in (
            ("box_threshold", self.box_threshold),
            ("text_threshold", self.text_threshold),
        ):
            if not 0.0 <= threshold <= 1.0:
                raise ConfigurationError(f"{name} must be between 0 and 1; got {threshold}.")


@dataclass(frozen=True, slots=True)
class PipelineSettings:
    """Filesystem, batching, device, and safety settings."""

    images_dir: Path = Path("images")
    db_path: Path = Path("qdrant_fashion_db")
    collection_name: str = COLLECTION_NAME
    recursive: bool = True
    batch_size: int = 8
    max_image_pixels: int = 100_000_000
    max_image_dimension: int = 2_048
    device: str = "auto"
    precision: str = "auto"
    recreate_collection: bool = False
    skip_existing: bool = True
    image_extensions: frozenset[str] = field(default_factory=lambda: DEFAULT_IMAGE_EXTENSIONS)

    def __post_init__(self) -> None:
        if not self.collection_name.strip():
            raise ConfigurationError("Collection name cannot be empty.")
        if self.batch_size < 1:
            raise ConfigurationError("batch_size must be at least 1.")
        if self.max_image_pixels < 1:
            raise ConfigurationError("max_image_pixels must be positive.")
        if self.max_image_dimension < 1:
            raise ConfigurationError("max_image_dimension must be positive.")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ConfigurationError(f"Unsupported device selection: {self.device!r}.")
        if self.precision not in {"auto", "float32", "float16", "bfloat16"}:
            raise ConfigurationError(f"Unsupported precision selection: {self.precision!r}.")
        if not self.image_extensions:
            raise ConfigurationError("At least one image extension is required.")


def build_index_manifest(
    model_settings: ModelSettings,
    pipeline_settings: PipelineSettings,
    *,
    compute_dtype: str,
) -> dict[str, Any]:
    """Describe every setting that can change the stored embedding space."""

    return {
        "manifest_version": 1,
        "pipeline_code_version": 1,
        "models": {
            "detector": {
                "model_id": model_settings.detector_model_id,
                "revision": model_settings.detector_revision,
            },
            "garment": {
                "model_id": model_settings.fashion_model_id,
                "revision": model_settings.fashion_revision,
            },
            "region": {
                "model_id": model_settings.region_model_id,
                "revision": model_settings.region_revision,
            },
            "vibe": {
                "model_id": model_settings.vibe_model_id,
                "revision": model_settings.vibe_revision,
            },
        },
        "garment_detection": {
            "labels": list(model_settings.garment_labels),
            "box_threshold": model_settings.box_threshold,
            "text_threshold": model_settings.text_threshold,
            "fallback": "largest_center_square",
        },
        "global_preprocessing": {
            "max_image_pixels": pipeline_settings.max_image_pixels,
            "max_image_dimension": pipeline_settings.max_image_dimension,
            "exif_transpose": True,
            "alpha_background": "white",
            "resize_filter": "lanczos",
        },
        "embedding_contract": {
            GARMENT_VECTOR_NAME: {
                "size": GARMENT_VECTOR_SIZE,
                "extractor": "clip_get_image_features",
                "normalization": "l2",
            },
            REGION_VECTOR_NAME: {
                "size": REGION_VECTOR_SIZE,
                "extractor": "clip_get_image_features",
                "normalization": "l2",
            },
            VIBE_VECTOR_NAME: {
                "size": VIBE_VECTOR_SIZE,
                "extractor": "open_clip_encode_image",
                "normalization": "l2",
            },
        },
        "compute_dtype": compute_dtype,
        "runtime_versions": {
            package: _package_version(package)
            for package in ("Pillow", "open-clip-torch", "torch", "torchvision", "transformers")
        },
    }


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"
