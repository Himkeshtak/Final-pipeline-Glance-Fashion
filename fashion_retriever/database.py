"""Read-only named-vector queries against the local fashion Qdrant collection."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fashion_indexer.config import (
    GARMENT_VECTOR_NAME,
    GARMENT_VECTOR_SIZE,
    REGION_VECTOR_NAME,
    REGION_VECTOR_SIZE,
    VIBE_VECTOR_NAME,
    VIBE_VECTOR_SIZE,
    ModelSettings,
)
from fashion_retriever.encoders import EncodedQueries
from fashion_retriever.errors import EmptyCollectionError, RetrievalDatabaseError


@dataclass(frozen=True, slots=True)
class BranchHit:
    """One raw Qdrant nearest-neighbor result in a named vector space."""

    id: str
    score: float
    payload: dict[str, Any]


class FashionSearchStore:
    """Validate and batch-query all three independent named vector indexes."""

    VECTOR_SIZES = {
        GARMENT_VECTOR_NAME: GARMENT_VECTOR_SIZE,
        REGION_VECTOR_NAME: REGION_VECTOR_SIZE,
        VIBE_VECTOR_NAME: VIBE_VECTOR_SIZE,
    }

    def __init__(
        self,
        path: Path,
        collection_name: str,
        *,
        client: Any | None = None,
        query_attempts: int = 3,
    ) -> None:
        self.path = Path(path)
        self.collection_name = collection_name
        self.query_attempts = max(1, query_attempts)
        collection_key = hashlib.sha256(collection_name.encode("utf-8")).hexdigest()[:12]
        storage_name = self.path.name or "qdrant"
        self.manifest_path = self.path.with_name(
            f"{storage_name}.{collection_key}.index-manifest.json"
        )
        if client is not None:
            self.client = client
            return
        try:
            from qdrant_client import QdrantClient

            self.client = QdrantClient(path=str(self.path))
        except Exception as exc:
            raise RetrievalDatabaseError(
                f"Cannot open local Qdrant database at {self.path.resolve(strict=False)}. "
                f"Ensure the indexer is not using the same local storage. Details: {exc}"
            ) from exc

    def validate_ready(self, model_settings: ModelSettings) -> int:
        """Validate collection schema, non-empty state, and model-space manifest."""

        try:
            if not self.client.collection_exists(self.collection_name):
                raise RetrievalDatabaseError(
                    f"Qdrant collection {self.collection_name!r} does not exist. "
                    "Run indexing first."
                )
            self._validate_schema()
            count = int(self.client.count(collection_name=self.collection_name, exact=True).count)
            if count == 0:
                raise EmptyCollectionError(
                    f"Qdrant collection {self.collection_name!r} contains no images. "
                    "Complete indexing before retrieval."
                )
            self._validate_manifest(model_settings)
            return count
        except (RetrievalDatabaseError, EmptyCollectionError):
            raise
        except Exception as exc:
            raise RetrievalDatabaseError(f"Failed to validate Qdrant collection: {exc}") from exc

    def search_all(
        self,
        vectors: EncodedQueries,
        *,
        limit: int,
    ) -> dict[str, list[BranchHit]]:
        """Execute the three named-vector searches in one Qdrant batch request."""

        vector_map = {
            GARMENT_VECTOR_NAME: vectors.garment_vector,
            REGION_VECTOR_NAME: vectors.region_vector,
            VIBE_VECTOR_NAME: vectors.vibe_vector,
        }
        models = self._models()
        requests = [
            models.QueryRequest(
                query=vector_map[name],
                using=name,
                limit=limit,
                with_payload=["image_path", "file_hash", "timestamp"],
                with_vector=False,
            )
            for name in self.VECTOR_SIZES
        ]

        last_error: Exception | None = None
        for attempt in range(1, self.query_attempts + 1):
            try:
                responses = self.client.query_batch_points(
                    collection_name=self.collection_name,
                    requests=requests,
                )
                return self._parse_batch_responses(responses)
            except RetrievalDatabaseError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt < self.query_attempts:
                    time.sleep(0.2 * (2 ** (attempt - 1)))
        raise RetrievalDatabaseError(
            f"Qdrant batch search failed after {self.query_attempts} attempts: {last_error}"
        ) from last_error

    def _parse_batch_responses(self, responses: Sequence[Any]) -> dict[str, list[BranchHit]]:
        names = list(self.VECTOR_SIZES)
        if len(responses) != len(names):
            raise RetrievalDatabaseError(
                f"Qdrant returned {len(responses)} branch responses; expected {len(names)}."
            )
        parsed: dict[str, list[BranchHit]] = {}
        for name, response in zip(names, responses, strict=True):
            points = getattr(response, "points", None)
            if points is None:
                raise RetrievalDatabaseError(f"Qdrant response for {name!r} has no points field.")
            branch_hits: list[BranchHit] = []
            for point in points:
                score = float(point.score)
                if not math.isfinite(score):
                    raise RetrievalDatabaseError(
                        f"Qdrant returned a non-finite score for point {point.id}."
                    )
                payload = dict(point.payload) if isinstance(point.payload, Mapping) else {}
                branch_hits.append(BranchHit(id=str(point.id), score=score, payload=payload))
            parsed[name] = branch_hits
        return parsed

    def _validate_schema(self) -> None:
        info = self.client.get_collection(self.collection_name)
        vectors = info.config.params.vectors
        if not isinstance(vectors, Mapping):
            raise RetrievalDatabaseError("The collection does not use named vectors.")
        if set(vectors) != set(self.VECTOR_SIZES):
            raise RetrievalDatabaseError(
                f"Collection vectors are {sorted(vectors)}; expected {sorted(self.VECTOR_SIZES)}."
            )
        for name, expected_size in self.VECTOR_SIZES.items():
            params = vectors[name]
            distance = str(getattr(params.distance, "value", params.distance)).lower()
            if int(params.size) != expected_size or not distance.endswith("cosine"):
                raise RetrievalDatabaseError(
                    f"Named vector {name!r} must be size {expected_size} with cosine distance."
                )

    def _validate_manifest(self, settings: ModelSettings) -> None:
        if not self.manifest_path.is_file():
            raise RetrievalDatabaseError(
                f"Index manifest is missing at {self.manifest_path}; model compatibility "
                "cannot be proven."
            )
        try:
            with self.manifest_path.open("r", encoding="utf-8") as source:
                manifest = json.load(source)
            index_manifest = manifest["index_manifest"]
            indexed_models = index_manifest["models"]
            embedding_contract = index_manifest["embedding_contract"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RetrievalDatabaseError(f"Index manifest is invalid: {exc}") from exc

        expected = {
            "garment": (settings.fashion_model_id, settings.fashion_revision),
            "region": (settings.region_model_id, settings.region_revision),
            "vibe": (settings.vibe_model_id, settings.vibe_revision),
        }
        for branch, (model_id, revision) in expected.items():
            entry = indexed_models.get(branch, {})
            if entry.get("model_id") != model_id or entry.get("revision") != revision:
                raise RetrievalDatabaseError(
                    f"The {branch} text encoder does not match the indexed image model. "
                    "Reindex with the current model manifest before retrieval."
                )

        expected_contract = {
            GARMENT_VECTOR_NAME: (GARMENT_VECTOR_SIZE, "clip_get_image_features"),
            REGION_VECTOR_NAME: (REGION_VECTOR_SIZE, "clip_get_image_features"),
            VIBE_VECTOR_NAME: (VIBE_VECTOR_SIZE, "open_clip_encode_image"),
        }
        for name, (size, extractor) in expected_contract.items():
            entry = embedding_contract.get(name, {})
            if (
                entry.get("size") != size
                or entry.get("extractor") != extractor
                or entry.get("normalization") != "l2"
            ):
                raise RetrievalDatabaseError(
                    f"The indexed embedding contract for {name!r} is incompatible with "
                    "text retrieval. Reindex with the current pipeline."
                )

    def close(self) -> None:
        close_method = getattr(self.client, "close", None)
        if callable(close_method):
            close_method()

    def __enter__(self) -> FashionSearchStore:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    @staticmethod
    def _models() -> Any:
        try:
            from qdrant_client import models

            return models
        except ImportError as exc:
            raise RetrievalDatabaseError("qdrant-client is not installed.") from exc
