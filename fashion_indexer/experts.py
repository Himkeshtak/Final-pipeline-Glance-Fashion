"""The detector plus three independent fashion image embedding branches."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from PIL import Image

from fashion_indexer.config import (
    GARMENT_VECTOR_SIZE,
    REGION_VECTOR_SIZE,
    VIBE_VECTOR_SIZE,
    ModelSettings,
)
from fashion_indexer.device import (
    DeviceSpec,
    move_batch_to_device,
    transformers_dtype_kwargs,
)
from fashion_indexer.errors import ModelInferenceError, ModelInitializationError

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class NamedEmbeddings:
    """The three unit-normalized vectors stored on one Qdrant point."""

    garment_vector: list[float]
    region_vector: list[float]
    vibe_vector: list[float]

    def __post_init__(self) -> None:
        validate_vector(self.garment_vector, GARMENT_VECTOR_SIZE, "garment_vector")
        validate_vector(self.region_vector, REGION_VECTOR_SIZE, "region_vector")
        validate_vector(self.vibe_vector, VIBE_VECTOR_SIZE, "vibe_vector")


def validate_vector(vector: Sequence[float], expected_size: int, name: str) -> None:
    """Validate vector shape, values, and the explicit unit-norm contract."""

    if len(vector) != expected_size:
        raise ModelInferenceError(f"{name} has dimension {len(vector)}; expected {expected_size}.")
    if not all(math.isfinite(float(value)) for value in vector):
        raise ModelInferenceError(f"{name} contains NaN or infinite values.")
    norm = math.sqrt(sum(float(value) ** 2 for value in vector))
    if not math.isclose(norm, 1.0, rel_tol=1e-4, abs_tol=1e-4):
        raise ModelInferenceError(f"{name} has L2 norm {norm:.8f}; expected 1.0.")


def tensor_to_unit_vector(features: torch.Tensor, expected_size: int, name: str) -> list[float]:
    """Convert a single model embedding tensor to a finite float32 unit vector."""

    if features.ndim == 1:
        features = features.unsqueeze(0)
    if features.ndim != 2 or features.shape[0] != 1:
        raise ModelInferenceError(
            f"{name} produced shape {tuple(features.shape)}; expected (1, {expected_size})."
        )
    if features.shape[1] != expected_size:
        raise ModelInferenceError(
            f"{name} produced dimension {features.shape[1]}; expected {expected_size}."
        )

    float_features = features.detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(float_features).all():
        raise ModelInferenceError(f"{name} produced NaN or infinite values.")
    norms = torch.linalg.vector_norm(float_features, ord=2, dim=-1)
    if torch.any(norms <= 1e-12):
        raise ModelInferenceError(f"{name} produced a zero-length embedding.")

    normalized = functional.normalize(float_features, p=2, dim=-1)
    vector = normalized[0].tolist()
    validate_vector(vector, expected_size, name)
    return vector


def center_square_crop(image: Image.Image) -> Image.Image:
    """Return the largest centered square without resizing the source."""

    side = min(image.width, image.height)
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    return image.crop((left, top, left + side, top + side))


def sanitize_box(
    box: Sequence[float], image_size: tuple[int, int]
) -> tuple[int, int, int, int] | None:
    """Clamp a floating xyxy detector box to valid PIL crop coordinates."""

    if len(box) != 4 or not all(math.isfinite(float(value)) for value in box):
        return None
    width, height = image_size
    left = max(0, min(width, math.floor(float(box[0]))))
    top = max(0, min(height, math.floor(float(box[1]))))
    right = max(0, min(width, math.ceil(float(box[2]))))
    bottom = max(0, min(height, math.ceil(float(box[3]))))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


class GarmentCropDetector:
    """Open-vocabulary Grounding DINO detector selecting one best garment box."""

    def __init__(self, settings: ModelSettings, device_spec: DeviceSpec) -> None:
        self.settings = settings
        self.device_spec = device_spec
        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            load_kwargs = _transformer_load_kwargs(
                settings.cache_dir, settings.detector_revision, device_spec
            )
            self.processor = AutoProcessor.from_pretrained(
                settings.detector_model_id,
                **_hub_kwargs(settings.cache_dir, settings.detector_revision),
            )
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                settings.detector_model_id, **load_kwargs
            )
            self.model.to(device_spec.device)
            self.model.eval()
            self.model.requires_grad_(False)
        except Exception as exc:
            raise ModelInitializationError(
                f"Failed to load garment detector {settings.detector_model_id!r}: {exc}"
            ) from exc

    def crop(self, image: Image.Image) -> Image.Image:
        labels = [list(self.settings.garment_labels)]
        try:
            encoded = self.processor(images=image, text=labels, return_tensors="pt")
            inputs = move_batch_to_device(encoded, self.device_spec)
            with torch.inference_mode():
                outputs = self.model(**inputs)
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                input_ids=inputs.get("input_ids"),
                threshold=self.settings.box_threshold,
                text_threshold=self.settings.text_threshold,
                target_sizes=[(image.height, image.width)],
            )
        except Exception as exc:
            raise ModelInferenceError(f"Garment detection failed: {exc}") from exc

        if not results:
            LOGGER.debug("Detector returned no result object; using center crop.")
            return center_square_crop(image)

        scores = results[0].get("scores")
        boxes = results[0].get("boxes")
        if scores is None or boxes is None or len(scores) == 0 or len(boxes) == 0:
            LOGGER.debug("Detector returned no garment boxes; using center crop.")
            return center_square_crop(image)

        best_index = int(torch.as_tensor(scores).argmax().item())
        best_box = torch.as_tensor(boxes[best_index]).detach().cpu().tolist()
        crop_box = sanitize_box(best_box, image.size)
        if crop_box is None:
            LOGGER.debug("Detector returned an invalid best box; using center crop.")
            return center_square_crop(image)
        return image.crop(crop_box)


class FashionClipEncoder:
    """Fashion-CLIP projected image encoder producing 512-dimensional vectors."""

    def __init__(self, settings: ModelSettings, device_spec: DeviceSpec) -> None:
        self.device_spec = device_spec
        try:
            from transformers import AutoProcessor, CLIPModel

            self.processor = AutoProcessor.from_pretrained(
                settings.fashion_model_id,
                **_hub_kwargs(settings.cache_dir, settings.fashion_revision),
            )
            self.model = CLIPModel.from_pretrained(
                settings.fashion_model_id,
                **_transformer_load_kwargs(
                    settings.cache_dir, settings.fashion_revision, device_spec
                ),
            )
            projection_size = int(self.model.config.projection_dim)
            if projection_size != GARMENT_VECTOR_SIZE:
                raise ValueError(
                    f"checkpoint projection_dim is {projection_size}, not {GARMENT_VECTOR_SIZE}"
                )
            self.model.to(device_spec.device)
            self.model.eval()
            self.model.requires_grad_(False)
        except Exception as exc:
            raise ModelInitializationError(
                f"Failed to load Fashion-CLIP {settings.fashion_model_id!r}: {exc}"
            ) from exc

    def encode(self, garment_image: Image.Image) -> list[float]:
        try:
            encoded = self.processor(images=garment_image, return_tensors="pt")
            inputs = move_batch_to_device(encoded, self.device_spec)
            with torch.inference_mode():
                features = self.model.get_image_features(pixel_values=inputs["pixel_values"])
            return tensor_to_unit_vector(
                _coerce_feature_tensor(features), GARMENT_VECTOR_SIZE, "garment_vector"
            )
        except ModelInferenceError:
            raise
        except Exception as exc:
            raise ModelInferenceError(f"Fashion-CLIP inference failed: {exc}") from exc


class RegionSceneEncoder:
    """Full-image paired CLIP encoder producing 768-dimensional scene vectors."""

    def __init__(self, settings: ModelSettings, device_spec: DeviceSpec) -> None:
        self.device_spec = device_spec
        try:
            from transformers import AutoProcessor, CLIPModel

            self.processor = AutoProcessor.from_pretrained(
                settings.region_model_id,
                **_hub_kwargs(settings.cache_dir, settings.region_revision),
            )
            self.model = CLIPModel.from_pretrained(
                settings.region_model_id,
                **_transformer_load_kwargs(
                    settings.cache_dir, settings.region_revision, device_spec
                ),
            )
            projection_size = int(self.model.config.projection_dim)
            if projection_size != REGION_VECTOR_SIZE:
                raise ValueError(
                    f"checkpoint projection_dim is {projection_size}, not {REGION_VECTOR_SIZE}"
                )
            self.model.to(device_spec.device)
            self.model.eval()
            self.model.requires_grad_(False)
        except Exception as exc:
            raise ModelInitializationError(
                f"Failed to load region ViT {settings.region_model_id!r}: {exc}"
            ) from exc

    def encode(self, full_image: Image.Image) -> list[float]:
        try:
            encoded = self.processor(images=full_image, return_tensors="pt")
            inputs = move_batch_to_device(encoded, self.device_spec)
            with torch.inference_mode():
                features = self.model.get_image_features(pixel_values=inputs["pixel_values"])
            return tensor_to_unit_vector(
                _coerce_feature_tensor(features), REGION_VECTOR_SIZE, "region_vector"
            )
        except ModelInferenceError:
            raise
        except Exception as exc:
            raise ModelInferenceError(f"Region CLIP inference failed: {exc}") from exc


class VibeAestheticEncoder:
    """Full-image OpenCLIP global encoder producing 768-dimensional vibe vectors."""

    def __init__(self, settings: ModelSettings, device_spec: DeviceSpec) -> None:
        self.device_spec = device_spec
        try:
            import open_clip
            from huggingface_hub import snapshot_download

            snapshot_kwargs: dict[str, Any] = {
                "repo_id": settings.vibe_model_id,
                "revision": settings.vibe_revision,
                "allow_patterns": [
                    "open_clip_config.json",
                    "open_clip_model.safetensors",
                    "open_clip_pytorch_model.bin",
                    "open_clip_pytorch_model.safetensors",
                ],
            }
            if settings.cache_dir is not None:
                snapshot_kwargs["cache_dir"] = str(settings.cache_dir)
            snapshot_path = snapshot_download(**snapshot_kwargs)
            pinned_model_source = f"local-dir:{snapshot_path}"

            kwargs: dict[str, Any] = {
                "device": device_spec.device,
                "precision": _open_clip_precision(device_spec.dtype),
            }
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                pinned_model_source, **kwargs
            )
            reported_size = getattr(self.model.visual, "output_dim", None)
            if reported_size is not None and int(reported_size) != VIBE_VECTOR_SIZE:
                raise ValueError(
                    f"checkpoint output_dim is {reported_size}, not {VIBE_VECTOR_SIZE}"
                )
            self.model.eval()
            self.model.requires_grad_(False)
        except Exception as exc:
            raise ModelInitializationError(
                f"Failed to load OpenCLIP vibe model {settings.vibe_model_id!r}: {exc}"
            ) from exc

    def encode(self, full_image: Image.Image) -> list[float]:
        try:
            pixel_values = self.preprocess(full_image).unsqueeze(0)
            pixel_values = pixel_values.to(
                device=self.device_spec.device, dtype=self.device_spec.dtype
            )
            with torch.inference_mode():
                features = self.model.encode_image(pixel_values)
            return tensor_to_unit_vector(features, VIBE_VECTOR_SIZE, "vibe_vector")
        except ModelInferenceError:
            raise
        except Exception as exc:
            raise ModelInferenceError(f"OpenCLIP vibe inference failed: {exc}") from exc


class MultiExpertEncoder:
    """Orchestrate the three independent image branches for one source image."""

    def __init__(self, settings: ModelSettings, device_spec: DeviceSpec) -> None:
        LOGGER.info("Loading Grounding DINO garment crop detector")
        self.detector = GarmentCropDetector(settings, device_spec)
        LOGGER.info("Loading Fashion-CLIP garment encoder")
        self.garment_encoder = FashionClipEncoder(settings, device_spec)
        LOGGER.info("Loading full-image region CLIP encoder")
        self.region_encoder = RegionSceneEncoder(settings, device_spec)
        LOGGER.info("Loading full-image OpenCLIP vibe encoder")
        self.vibe_encoder = VibeAestheticEncoder(settings, device_spec)

    def encode(self, full_image: Image.Image) -> NamedEmbeddings:
        garment_crop = self.detector.crop(full_image)
        try:
            garment_vector = self.garment_encoder.encode(garment_crop)
        finally:
            garment_crop.close()

        region_vector = self.region_encoder.encode(full_image)
        vibe_vector = self.vibe_encoder.encode(full_image)
        return NamedEmbeddings(
            garment_vector=garment_vector,
            region_vector=region_vector,
            vibe_vector=vibe_vector,
        )


def _coerce_feature_tensor(features: Any) -> torch.Tensor:
    if isinstance(features, torch.Tensor):
        return features
    for attribute in ("image_embeds", "pooler_output"):
        candidate = getattr(features, attribute, None)
        if isinstance(candidate, torch.Tensor):
            return candidate
    raise ModelInferenceError(f"Model returned unsupported feature type {type(features).__name__}.")


def _hub_kwargs(cache_dir: Path | None, revision: str) -> dict[str, str]:
    kwargs = {"revision": revision}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    return kwargs


def _transformer_load_kwargs(
    cache_dir: Path | None, revision: str, device_spec: DeviceSpec
) -> dict[str, Any]:
    kwargs: dict[str, Any] = _hub_kwargs(cache_dir, revision)
    kwargs.update(transformers_dtype_kwargs(device_spec))
    return kwargs


def _open_clip_precision(dtype: torch.dtype) -> str:
    precision_by_dtype = {
        torch.float32: "fp32",
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
    }
    try:
        return precision_by_dtype[dtype]
    except KeyError as exc:
        raise ModelInitializationError(f"OpenCLIP does not support dtype {dtype}.") from exc
