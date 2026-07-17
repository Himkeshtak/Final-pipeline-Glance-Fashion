from __future__ import annotations

import pytest
import torch

from fashion_indexer.device import resolve_device
from fashion_indexer.errors import DeviceSelectionError


def test_explicit_cpu_float32_selection() -> None:
    spec = resolve_device("cpu", "float32")

    assert spec.device.type == "cpu"
    assert spec.dtype == torch.float32


def test_float16_cpu_is_rejected() -> None:
    with pytest.raises(DeviceSelectionError, match="float16 on CPU"):
        resolve_device("cpu", "float16")


def test_unavailable_cuda_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(DeviceSelectionError, match="CUDA was requested"):
        resolve_device("cuda", "float32")
