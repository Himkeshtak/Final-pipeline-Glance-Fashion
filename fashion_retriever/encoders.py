"""Three model-matched text experts for Strategy A named-vector querying."""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

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
from fashion_indexer.experts import tensor_to_unit_vector, validate_vector
from fashion_retriever.errors import TextEncodingError
from fashion_retriever.schemas import QueryPlan

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EncodedQueries:
    """Three unit-normalized text vectors matched to their image index spaces."""

    garment_vector: list[float]
    region_vector: list[float]
    vibe_vector: list[float]

    def __post_init__(self) -> None:
        validate_vector(self.garment_vector, GARMENT_VECTOR_SIZE, "garment query vector")
        validate_vector(self.region_vector, REGION_VECTOR_SIZE, "region query vector")
        validate_vector(self.vibe_vector, VIBE_VECTOR_SIZE, "vibe query vector")


class TransformersClipTextEncoder:
    """A Hugging Face CLIP text tower with strict projection validation."""

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        expected_size: int,
        vector_name: str,
        cache_dir: Path | None,
        device_spec: DeviceSpec,
    ) -> None:
        self.expected_size = expected_size
        self.vector_name = vector_name
        self.device_spec = device_spec
        try:
            from transformers import AutoProcessor, CLIPModel

            hub_kwargs = _hub_kwargs(cache_dir, revision)
            self.processor = AutoProcessor.from_pretrained(model_id, **hub_kwargs)
            load_kwargs: dict[str, Any] = dict(hub_kwargs)
            load_kwargs.update(transformers_dtype_kwargs(device_spec))
            self.model = CLIPModel.from_pretrained(model_id, **load_kwargs)
            projection_size = int(self.model.config.projection_dim)
            if projection_size != expected_size:
                raise ValueError(
                    f"checkpoint projection_dim is {projection_size}, expected {expected_size}"
                )
            self.model.to(device_spec.device)
            self.model.eval()
            self.model.requires_grad_(False)
        except Exception as exc:
            raise TextEncodingError(f"Failed to load text encoder {model_id!r}: {exc}") from exc

    def encode(self, text: str) -> list[float]:
        query = _validate_query_text(text, self.vector_name)
        try:
            encoded = self.processor(
                text=[query],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = move_batch_to_device(encoded, self.device_spec)
            text_inputs = {
                key: value
                for key, value in inputs.items()
                if key in {"input_ids", "attention_mask", "position_ids"}
            }
            with torch.inference_mode():
                features = self.model.get_text_features(**text_inputs)
            return tensor_to_unit_vector(
                _coerce_text_feature_tensor(features),
                self.expected_size,
                self.vector_name,
            )
        except Exception as exc:
            if isinstance(exc, TextEncodingError):
                raise
            raise TextEncodingError(f"{self.vector_name} text encoding failed: {exc}") from exc


class OpenClipTextEncoder:
    """The exact OpenCLIP text tower paired with the indexed vibe image tower."""

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
            model_source = f"local-dir:{snapshot_path}"
            self.model, _, _ = open_clip.create_model_and_transforms(
                model_source,
                device=device_spec.device,
                precision=_open_clip_precision(device_spec.dtype),
            )
            self.tokenizer = open_clip.get_tokenizer(model_source)
            text_tower = getattr(self.model, "text", None)
            output_size = getattr(text_tower, "output_dim", None)
            if output_size is None:
                output_size = getattr(self.model, "embed_dim", None)
            if output_size is not None and int(output_size) != VIBE_VECTOR_SIZE:
                raise ValueError(
                    f"checkpoint text output dimension is {output_size}, "
                    f"expected {VIBE_VECTOR_SIZE}"
                )
            self.model.eval()
            self.model.requires_grad_(False)
        except Exception as exc:
            raise TextEncodingError(
                f"Failed to load OpenCLIP vibe text encoder {settings.vibe_model_id!r}: {exc}"
            ) from exc

    def encode(self, text: str) -> list[float]:
        query = _validate_query_text(text, "vibe query vector")
        try:
            tokens = self.tokenizer([query]).to(device=self.device_spec.device)
            with torch.inference_mode():
                features = self.model.encode_text(tokens)
            return tensor_to_unit_vector(features, VIBE_VECTOR_SIZE, "vibe query vector")
        except Exception as exc:
            raise TextEncodingError(f"Vibe text encoding failed: {exc}") from exc


class MultiExpertTextEncoder:
    """Own and execute all three model-matched text expert branches."""

    def __init__(
        self,
        settings: ModelSettings,
        device_spec: DeviceSpec,
        *,
        parallel: bool = True,
    ) -> None:
        self.parallel = parallel
        LOGGER.info("Loading Fashion-CLIP garment text encoder")
        self.garment_encoder = TransformersClipTextEncoder(
            model_id=settings.fashion_model_id,
            revision=settings.fashion_revision,
            expected_size=GARMENT_VECTOR_SIZE,
            vector_name="garment query vector",
            cache_dir=settings.cache_dir,
            device_spec=device_spec,
        )
        LOGGER.info("Loading paired CLIP region text encoder")
        self.region_encoder = TransformersClipTextEncoder(
            model_id=settings.region_model_id,
            revision=settings.region_revision,
            expected_size=REGION_VECTOR_SIZE,
            vector_name="region query vector",
            cache_dir=settings.cache_dir,
            device_spec=device_spec,
        )
        LOGGER.info("Loading OpenCLIP vibe text encoder")
        self.vibe_encoder = OpenClipTextEncoder(settings, device_spec)

    def encode(self, plan: QueryPlan) -> EncodedQueries:
        jobs: dict[str, tuple[Callable[[str], list[float]], str]] = {
            "garment_vector": (self.garment_encoder.encode, plan.garment_query),
            "region_vector": (self.region_encoder.encode, plan.region_query),
            "vibe_vector": (self.vibe_encoder.encode, plan.vibe_query),
        }
        if not self.parallel:
            vectors = {name: encoder(text) for name, (encoder, text) in jobs.items()}
        else:
            vectors = self._encode_parallel(jobs)
        return EncodedQueries(
            garment_vector=vectors["garment_vector"],
            region_vector=vectors["region_vector"],
            vibe_vector=vectors["vibe_vector"],
        )

    @staticmethod
    def _encode_parallel(
        jobs: dict[str, tuple[Callable[[str], list[float]], str]],
    ) -> dict[str, list[float]]:
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="fashion-text") as executor:
            futures: dict[str, Future[list[float]]] = {
                name: executor.submit(encoder, text) for name, (encoder, text) in jobs.items()
            }
            try:
                return {name: future.result() for name, future in futures.items()}
            except Exception:
                for future in futures.values():
                    future.cancel()
                raise


def _validate_query_text(text: str, vector_name: str) -> str:
    query = text.strip()
    if not query:
        raise TextEncodingError(f"{vector_name} cannot be encoded from empty text.")
    return query


def _coerce_text_feature_tensor(features: Any) -> torch.Tensor:
    if isinstance(features, torch.Tensor):
        return features
    for attribute in ("text_embeds", "pooler_output"):
        candidate = getattr(features, attribute, None)
        if isinstance(candidate, torch.Tensor):
            return candidate
    raise TextEncodingError(
        f"Text model returned unsupported feature type {type(features).__name__}."
    )


def _hub_kwargs(cache_dir: Path | None, revision: str) -> dict[str, str]:
    kwargs = {"revision": revision}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    return kwargs


def _open_clip_precision(dtype: torch.dtype) -> str:
    try:
        return {
            torch.float32: "fp32",
            torch.float16: "fp16",
            torch.bfloat16: "bf16",
        }[dtype]
    except KeyError as exc:
        raise TextEncodingError(f"OpenCLIP does not support dtype {dtype}.") from exc
