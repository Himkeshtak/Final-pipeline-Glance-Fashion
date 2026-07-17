from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from PIL import Image

from fashion_indexer.errors import DuplicateImageError, ImageDirectoryError, InvalidImageError
from fashion_indexer.preprocessing import ImagePreprocessor


def _save_image(
    path: Path,
    size: tuple[int, int] = (32, 24),
    color: tuple[int, int, int, int] = (20, 40, 60, 255),
) -> None:
    image = Image.new("RGBA", size, color)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        image = image.convert("RGB")
    image.save(path)


def test_discovery_is_recursive_filtered_and_deterministic(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    _save_image(tmp_path / "b.png")
    _save_image(nested / "A.JPG")
    (tmp_path / "notes.txt").write_text("not an image", encoding="utf-8")

    paths = ImagePreprocessor(tmp_path, recursive=True).discover()

    assert [path.name for path in paths] == ["b.png", "A.JPG"]


def test_missing_input_directory_is_explicit(tmp_path: Path) -> None:
    with pytest.raises(ImageDirectoryError, match="does not exist"):
        ImagePreprocessor(tmp_path / "missing").discover()


def test_md5_uses_raw_bytes_and_duplicate_is_rejected(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _save_image(first)
    second.write_bytes(first.read_bytes())
    expected_hash = hashlib.md5(first.read_bytes()).hexdigest()
    preprocessor = ImagePreprocessor(tmp_path)

    prepared = preprocessor.prepare(first)
    try:
        assert prepared.file_hash == expected_hash
        assert prepared.image.mode == "RGB"
    finally:
        prepared.close()

    with pytest.raises(DuplicateImageError):
        preprocessor.prepare(second)


def test_corrupt_image_is_rejected(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(b"this is not jpeg data")

    with pytest.raises(InvalidImageError, match="Invalid or corrupt"):
        ImagePreprocessor(tmp_path).prepare(corrupt)


def test_global_resize_preserves_aspect_ratio(tmp_path: Path) -> None:
    source = tmp_path / "wide.png"
    _save_image(source, size=(400, 100))

    prepared = ImagePreprocessor(tmp_path, max_image_dimension=200).prepare(source)
    try:
        assert prepared.image.size == (200, 50)
    finally:
        prepared.close()
