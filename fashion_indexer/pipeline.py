"""End-to-end image preparation, expert inference, and batched Qdrant indexing."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from PIL import Image
from tqdm import tqdm

from fashion_indexer.database import FashionVectorStore
from fashion_indexer.errors import (
    DatabaseError,
    DuplicateImageError,
    InvalidImageError,
    ModelInferenceError,
)
from fashion_indexer.experts import NamedEmbeddings
from fashion_indexer.preprocessing import ImagePreprocessor

LOGGER = logging.getLogger(__name__)


class ExpertEncoder(Protocol):
    def encode(self, full_image: Image.Image) -> NamedEmbeddings: ...


@dataclass(slots=True)
class IndexingStats:
    discovered: int = 0
    indexed: int = 0
    duplicate_files: int = 0
    already_indexed: int = 0
    invalid_images: int = 0
    inference_failures: int = 0

    @property
    def skipped(self) -> int:
        return (
            self.duplicate_files
            + self.already_indexed
            + self.invalid_images
            + self.inference_failures
        )


class FashionIndexingPipeline:
    """Process each source once and atomically upsert completed multi-vector points."""

    def __init__(
        self,
        preprocessor: ImagePreprocessor,
        encoder: ExpertEncoder,
        store: FashionVectorStore,
        *,
        batch_size: int = 8,
        existing_hashes: set[str] | None = None,
    ) -> None:
        self.preprocessor = preprocessor
        self.encoder = encoder
        self.store = store
        self.batch_size = batch_size
        self.existing_hashes = existing_hashes if existing_hashes is not None else set()

    def run(self, paths: Sequence[Path] | None = None) -> IndexingStats:
        image_paths = list(paths) if paths is not None else self.preprocessor.discover()
        stats = IndexingStats(discovered=len(image_paths))
        pending_points: list[Any] = []
        pending_hashes: list[str] = []

        progress = tqdm(
            image_paths,
            desc="Indexing fashion images",
            unit="image",
            dynamic_ncols=True,
        )
        for path in progress:
            prepared = None
            try:
                prepared = self.preprocessor.prepare(path)
                if prepared.file_hash in self.existing_hashes:
                    stats.already_indexed += 1
                    continue

                embeddings = self.encoder.encode(prepared.image)
                point = self.store.make_point(
                    embeddings,
                    image_path=str(prepared.path),
                    file_hash=prepared.file_hash,
                )
                pending_points.append(point)
                pending_hashes.append(prepared.file_hash)
                if len(pending_points) >= self.batch_size:
                    self._flush(pending_points, pending_hashes, stats)
            except DuplicateImageError as exc:
                stats.duplicate_files += 1
                LOGGER.warning("Skipping duplicate image %s: %s", path, exc)
            except InvalidImageError as exc:
                stats.invalid_images += 1
                LOGGER.warning("Skipping invalid image %s: %s", path, exc)
            except DatabaseError:
                raise
            except ModelInferenceError as exc:
                stats.inference_failures += 1
                if prepared is not None:
                    self.preprocessor.release_hash(prepared.file_hash)
                LOGGER.exception("Failed to encode %s: %s", path, exc)
            finally:
                if prepared is not None:
                    prepared.close()
                progress.set_postfix(
                    indexed=stats.indexed,
                    queued=len(pending_points),
                    skipped=stats.skipped,
                )

        self._flush(pending_points, pending_hashes, stats)
        return stats

    def _flush(
        self,
        pending_points: list[Any],
        pending_hashes: list[str],
        stats: IndexingStats,
    ) -> None:
        if not pending_points:
            return
        self.store.upsert(pending_points)
        stats.indexed += len(pending_points)
        self.existing_hashes.update(pending_hashes)
        pending_points.clear()
        pending_hashes.clear()
