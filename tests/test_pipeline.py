from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from fashion_indexer.errors import DatabaseError, ModelInferenceError
from fashion_indexer.experts import NamedEmbeddings
from fashion_indexer.pipeline import FashionIndexingPipeline
from fashion_indexer.preprocessing import ImagePreprocessor


def _unit(size: int, index: int) -> list[float]:
    vector = [0.0] * size
    vector[index] = 1.0
    return vector


class FakeEncoder:
    def encode(self, full_image: Image.Image) -> NamedEmbeddings:
        assert full_image.mode == "RGB"
        return NamedEmbeddings(_unit(512, 0), _unit(768, 1), _unit(768, 2))


class FakeStore:
    def __init__(self) -> None:
        self.upserted: list[dict] = []

    def make_point(
        self,
        embeddings: NamedEmbeddings,
        *,
        image_path: str,
        file_hash: str,
        timestamp: str | None = None,
    ) -> dict:
        return {"path": image_path, "hash": file_hash, "vectors": embeddings}

    def upsert(self, points: list[dict]) -> None:
        self.upserted.extend(points)


class FailingStore(FakeStore):
    def upsert(self, points: list[dict]) -> None:
        raise DatabaseError("local database is locked")


class ExplodingEncoder:
    def encode(self, full_image: Image.Image) -> NamedEmbeddings:
        raise AssertionError("encoder must not run")


class FlakyEncoder:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, full_image: Image.Image) -> NamedEmbeddings:
        self.calls += 1
        if self.calls == 1:
            raise ModelInferenceError("transient failure")
        return NamedEmbeddings(_unit(512, 0), _unit(768, 1), _unit(768, 2))


def test_pipeline_skips_duplicates_and_flushes_batches(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    third = tmp_path / "third.png"
    Image.new("RGB", (16, 16), "red").save(first)
    second.write_bytes(first.read_bytes())
    Image.new("RGB", (16, 16), "blue").save(third)

    preprocessor = ImagePreprocessor(tmp_path)
    store = FakeStore()
    pipeline = FashionIndexingPipeline(
        preprocessor,
        FakeEncoder(),
        store,  # type: ignore[arg-type]
        batch_size=1,
    )

    stats = pipeline.run([first, second, third])

    assert stats.discovered == 3
    assert stats.indexed == 2
    assert stats.duplicate_files == 1
    assert stats.inference_failures == 0
    assert len(store.upserted) == 2


def test_pipeline_does_not_swallow_database_failures(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (16, 16), "red").save(source)
    pipeline = FashionIndexingPipeline(
        ImagePreprocessor(tmp_path),
        FakeEncoder(),
        FailingStore(),  # type: ignore[arg-type]
        batch_size=1,
    )

    with pytest.raises(DatabaseError, match="locked"):
        pipeline.run([source])


def test_existing_hash_skips_inference(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (16, 16), "red").save(source)
    existing_hash = ImagePreprocessor.compute_md5(source)
    store = FakeStore()
    pipeline = FashionIndexingPipeline(
        ImagePreprocessor(tmp_path),
        ExplodingEncoder(),
        store,  # type: ignore[arg-type]
        existing_hashes={existing_hash},
    )

    stats = pipeline.run([source])

    assert stats.already_indexed == 1
    assert stats.indexed == 0
    assert store.upserted == []


def test_duplicate_can_retry_after_transient_inference_failure(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (16, 16), "red").save(first)
    second.write_bytes(first.read_bytes())
    encoder = FlakyEncoder()
    store = FakeStore()
    pipeline = FashionIndexingPipeline(
        ImagePreprocessor(tmp_path),
        encoder,
        store,  # type: ignore[arg-type]
        batch_size=1,
    )

    stats = pipeline.run([first, second])

    assert encoder.calls == 2
    assert stats.inference_failures == 1
    assert stats.duplicate_files == 0
    assert stats.indexed == 1
