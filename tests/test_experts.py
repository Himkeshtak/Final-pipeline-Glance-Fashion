from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from fashion_indexer.device import DeviceSpec
from fashion_indexer.errors import ModelInferenceError
from fashion_indexer.experts import (
    GarmentCropDetector,
    center_square_crop,
    sanitize_box,
    tensor_to_unit_vector,
)


class FakeDetectorProcessor:
    def __init__(self, scores: torch.Tensor, boxes: torch.Tensor) -> None:
        self.scores = scores
        self.boxes = boxes
        self.received_text = None

    def __call__(self, *, images, text, return_tensors):
        self.received_text = text
        assert return_tensors == "pt"
        return {
            "pixel_values": torch.zeros((1, 3, 4, 4)),
            "input_ids": torch.ones((1, 2), dtype=torch.long),
        }

    def post_process_grounded_object_detection(self, *args, **kwargs):
        return [{"scores": self.scores, "boxes": self.boxes}]


class FakeDetectorModel:
    def __call__(self, **kwargs):
        return object()


def _fake_detector(scores: torch.Tensor, boxes: torch.Tensor) -> GarmentCropDetector:
    detector = GarmentCropDetector.__new__(GarmentCropDetector)
    detector.settings = SimpleNamespace(
        garment_labels=("garment", "clothing", "apparel", "dress"),
        box_threshold=0.25,
        text_threshold=0.25,
    )
    detector.device_spec = DeviceSpec(torch.device("cpu"), torch.float32)
    detector.processor = FakeDetectorProcessor(scores, boxes)
    detector.model = FakeDetectorModel()
    return detector


def test_center_square_crop_geometry() -> None:
    image = Image.new("RGB", (10, 6))
    crop = center_square_crop(image)
    try:
        assert crop.size == (6, 6)
    finally:
        crop.close()
        image.close()


def test_detector_falls_back_to_center_crop_when_no_boxes() -> None:
    detector = _fake_detector(torch.empty(0), torch.empty((0, 4)))
    image = Image.new("RGB", (10, 6))

    crop = detector.crop(image)

    try:
        assert crop.size == (6, 6)
        assert detector.processor.received_text == [["garment", "clothing", "apparel", "dress"]]
    finally:
        crop.close()
        image.close()


def test_detector_selects_highest_confidence_box() -> None:
    detector = _fake_detector(
        torch.tensor([0.4, 0.8]),
        torch.tensor([[0.0, 0.0, 2.0, 2.0], [2.0, 1.0, 9.0, 5.0]]),
    )
    image = Image.new("RGB", (10, 6))

    crop = detector.crop(image)

    try:
        assert crop.size == (7, 4)
    finally:
        crop.close()
        image.close()


@pytest.mark.parametrize(
    ("box", "expected"),
    [
        ((-1.2, 2.1, 11.9, 9.2), (0, 2, 10, 10)),
        ((2.2, 3.8, 8.1, 9.01), (2, 3, 9, 10)),
        ((4.0, 4.0, 4.0, 8.0), None),
        ((float("nan"), 0.0, 1.0, 1.0), None),
    ],
)
def test_sanitize_box(box: tuple[float, ...], expected: tuple[int, ...] | None) -> None:
    assert sanitize_box(box, (10, 10)) == expected


def test_tensor_conversion_normalizes_and_validates_dimension() -> None:
    result = tensor_to_unit_vector(torch.tensor([[3.0, 4.0]]), 2, "test_vector")

    assert result == pytest.approx([0.6, 0.8])
    assert math.sqrt(sum(value**2 for value in result)) == pytest.approx(1.0)


def test_tensor_conversion_rejects_nonfinite_values() -> None:
    with pytest.raises(ModelInferenceError, match="NaN or infinite"):
        tensor_to_unit_vector(torch.tensor([[float("nan"), 1.0]]), 2, "test_vector")
