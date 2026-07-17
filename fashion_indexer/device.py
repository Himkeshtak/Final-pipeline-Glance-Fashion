"""Accelerator selection and tensor movement helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from fashion_indexer.errors import DeviceSelectionError


@dataclass(frozen=True, slots=True)
class DeviceSpec:
    """Resolved torch device and model compute dtype."""

    device: torch.device
    dtype: torch.dtype

    @property
    def description(self) -> str:
        return f"{self.device.type} ({str(self.dtype).removeprefix('torch.')})"


def resolve_device(requested: str = "auto", precision: str = "auto") -> DeviceSpec:
    """Resolve CUDA, MPS, or CPU and validate the requested precision."""

    if requested == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif _mps_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif requested == "cuda":
        if not torch.cuda.is_available():
            raise DeviceSelectionError(
                "CUDA was requested, but torch.cuda.is_available() is false."
            )
        device = torch.device("cuda")
    elif requested == "mps":
        if not _mps_available():
            raise DeviceSelectionError("MPS was requested, but the MPS backend is unavailable.")
        device = torch.device("mps")
    elif requested == "cpu":
        device = torch.device("cpu")
    else:
        raise DeviceSelectionError(f"Unknown device selection: {requested!r}.")

    dtype = _resolve_dtype(device, precision)
    return DeviceSpec(device=device, dtype=dtype)


def move_batch_to_device(batch: Mapping[str, Any], device_spec: DeviceSpec) -> dict[str, Any]:
    """Move tensors to the model device while preserving integer tensor dtypes."""

    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                moved[key] = value.to(device=device_spec.device, dtype=device_spec.dtype)
            else:
                moved[key] = value.to(device=device_spec.device)
        else:
            moved[key] = value
    return moved


def transformers_dtype_kwargs(device_spec: DeviceSpec) -> dict[str, torch.dtype]:
    """Return conservative loading kwargs compatible with Transformers 4.x."""

    if device_spec.dtype == torch.float32:
        return {}
    return {"torch_dtype": device_spec.dtype}


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available() and backend.is_built())


def _resolve_dtype(device: torch.device, precision: str) -> torch.dtype:
    if precision == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32

    dtype_by_name = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        dtype = dtype_by_name[precision]
    except KeyError as exc:
        raise DeviceSelectionError(f"Unknown precision: {precision!r}.") from exc

    if device.type == "cpu" and dtype == torch.float16:
        raise DeviceSelectionError(
            "float16 on CPU is not supported reliably by all required vision operators; "
            "use float32 or bfloat16."
        )
    if device.type == "mps" and dtype == torch.bfloat16:
        raise DeviceSelectionError("bfloat16 is not supported reliably by the MPS backend.")
    if device.type == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise DeviceSelectionError(
            "bfloat16 was requested, but this CUDA device lacks BF16 support."
        )
    return dtype
