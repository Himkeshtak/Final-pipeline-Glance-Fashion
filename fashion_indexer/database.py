"""Local Qdrant storage with strict Strategy A named-vector schema validation."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
import uuid
from collections.abc import Mapping, Sequence
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fashion_indexer.config import (
    GARMENT_VECTOR_NAME,
    GARMENT_VECTOR_SIZE,
    REGION_VECTOR_NAME,
    REGION_VECTOR_SIZE,
    VIBE_VECTOR_NAME,
    VIBE_VECTOR_SIZE,
)
from fashion_indexer.errors import (
    CollectionSchemaError,
    DatabaseConflictError,
    DatabaseError,
)
from fashion_indexer.experts import NamedEmbeddings, validate_vector

LOGGER = logging.getLogger(__name__)


class FashionVectorStore:
    """Own a local Qdrant client and enforce the three-vector collection contract."""

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
        upsert_attempts: int = 3,
        index_manifest: Mapping[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.collection_name = collection_name
        self.upsert_attempts = max(1, upsert_attempts)
        self.index_manifest = (
            _normalize_json(index_manifest) if index_manifest is not None else None
        )
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

            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(self.path))
        except ImportError as exc:
            raise DatabaseError(
                "qdrant-client is not installed. Run `pip install -r requirements.txt`."
            ) from exc
        except Exception as exc:
            raise DatabaseConflictError(
                f"Cannot open local Qdrant database at {self.path.resolve(strict=False)}. "
                f"Ensure another process is not using the same local storage. Details: {exc}"
            ) from exc

    def ensure_collection(self, *, recreate: bool = False) -> None:
        """Create the collection or verify every existing named-vector parameter."""

        models = self._models()
        try:
            exists = self.client.collection_exists(self.collection_name)
            created = False
            if recreate and exists:
                LOGGER.warning("Deleting existing collection %s", self.collection_name)
                self.client.delete_collection(self.collection_name)
                exists = False

            if not exists:
                vectors_config = {
                    name: models.VectorParams(size=size, distance=models.Distance.COSINE)
                    for name, size in self.VECTOR_SIZES.items()
                }
                try:
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=vectors_config,
                    )
                    created = True
                except Exception:
                    # A concurrent creator may have won the race. Validate its result.
                    if not self.client.collection_exists(self.collection_name):
                        raise

            self._validate_collection_schema()
            if self.index_manifest is not None:
                collection_is_empty = self.count() == 0
                self._validate_or_write_manifest(
                    may_replace=created or recreate or collection_is_empty
                )
        except CollectionSchemaError:
            raise
        except Exception as exc:
            raise DatabaseError(
                f"Failed to create or validate collection {self.collection_name!r}: {exc}"
            ) from exc

    def existing_file_hashes(self, *, page_size: int = 256) -> set[str]:
        """Load stored hashes so repeat runs do not create duplicate UUID points."""

        hashes: set[str] = set()
        offset: Any | None = None
        try:
            while True:
                points, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=None,
                    limit=page_size,
                    offset=offset,
                    with_payload=["file_hash"],
                    with_vectors=False,
                )
                for point in points:
                    payload = point.payload or {}
                    value = payload.get("file_hash")
                    if isinstance(value, str):
                        hashes.add(value)
                if next_offset is None:
                    break
                if next_offset == offset:
                    raise DatabaseError("Qdrant scroll returned a repeated offset.")
                offset = next_offset
            return hashes
        except DatabaseError:
            raise
        except Exception as exc:
            raise DatabaseError(f"Failed to read existing image hashes: {exc}") from exc

    def make_point(
        self,
        embeddings: NamedEmbeddings,
        *,
        image_path: str,
        file_hash: str,
        timestamp: str | None = None,
    ) -> Any:
        """Build a UUID-v4 Qdrant point with exactly three named vectors and payload fields."""

        vector_map = {
            GARMENT_VECTOR_NAME: embeddings.garment_vector,
            REGION_VECTOR_NAME: embeddings.region_vector,
            VIBE_VECTOR_NAME: embeddings.vibe_vector,
        }
        self._validate_vector_map(vector_map)
        point_id = str(uuid.uuid4())
        payload = {
            "image_path": str(image_path),
            "file_hash": str(file_hash),
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        }
        return self._models().PointStruct(id=point_id, vector=vector_map, payload=payload)

    def upsert(self, points: Sequence[Any]) -> None:
        """Synchronously upsert a batch with bounded retries for transient local conflicts."""

        if not points:
            return
        last_error: Exception | None = None
        for attempt in range(1, self.upsert_attempts + 1):
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=list(points),
                    wait=True,
                )
                return
            except Exception as exc:
                last_error = exc
                if attempt < self.upsert_attempts:
                    delay = 0.25 * (2 ** (attempt - 1))
                    LOGGER.warning(
                        "Qdrant upsert attempt %d/%d failed; retrying in %.2fs: %s",
                        attempt,
                        self.upsert_attempts,
                        delay,
                        exc,
                    )
                    time.sleep(delay)
        raise DatabaseError(
            f"Qdrant upsert failed after {self.upsert_attempts} attempts: {last_error}"
        ) from last_error

    def count(self) -> int:
        try:
            result = self.client.count(collection_name=self.collection_name, exact=True)
            return int(result.count)
        except Exception as exc:
            raise DatabaseError(f"Failed to count indexed points: {exc}") from exc

    def close(self) -> None:
        close_method = getattr(self.client, "close", None)
        if callable(close_method):
            close_method()

    def __enter__(self) -> FashionVectorStore:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _validate_collection_schema(self) -> None:
        try:
            info = self.client.get_collection(self.collection_name)
            vectors = info.config.params.vectors
        except Exception as exc:
            raise DatabaseError(f"Cannot inspect collection schema: {exc}") from exc

        if not isinstance(vectors, Mapping):
            raise CollectionSchemaError(
                f"Collection {self.collection_name!r} does not use named vectors."
            )
        actual_names = set(vectors)
        expected_names = set(self.VECTOR_SIZES)
        if actual_names != expected_names:
            raise CollectionSchemaError(
                f"Collection named vectors are {sorted(actual_names)}; "
                f"expected exactly {sorted(expected_names)}. Use --recreate-collection only "
                "if deleting the existing collection is intentional."
            )

        for name, expected_size in self.VECTOR_SIZES.items():
            params = vectors[name]
            actual_size = int(params.size)
            actual_distance = _distance_name(params.distance)
            if actual_size != expected_size or actual_distance != "cosine":
                raise CollectionSchemaError(
                    f"Named vector {name!r} is size={actual_size}, distance={actual_distance}; "
                    f"expected size={expected_size}, distance=cosine."
                )
            if getattr(params, "datatype", None) is not None:
                raise CollectionSchemaError(
                    f"Named vector {name!r} has an explicit datatype; expected the default "
                    "float32 dense vector storage."
                )
            if getattr(params, "multivector_config", None) is not None:
                raise CollectionSchemaError(
                    f"Named vector {name!r} is configured as a multivector matrix; expected "
                    "one plain dense vector per point."
                )

    def _validate_or_write_manifest(self, *, may_replace: bool) -> None:
        expected = {
            "collection_name": self.collection_name,
            "index_manifest": self.index_manifest,
        }
        if self.manifest_path.exists():
            try:
                with self.manifest_path.open("r", encoding="utf-8") as source:
                    actual = json.load(source)
            except (OSError, json.JSONDecodeError) as exc:
                if not may_replace:
                    raise CollectionSchemaError(
                        f"Cannot read index manifest {self.manifest_path}: {exc}"
                    ) from exc
            else:
                if actual == expected:
                    return
                if not may_replace:
                    raise CollectionSchemaError(
                        "The existing collection was built with different model revisions, "
                        "preprocessing, precision, or runtime versions. Use a new database path "
                        "or --recreate-collection; mixing embedding spaces is refused."
                    )
        elif not may_replace:
            raise CollectionSchemaError(
                f"Collection {self.collection_name!r} contains points but has no compatible "
                f"index manifest at {self.manifest_path}. Refusing to append unverifiable vectors."
            )

        self._write_manifest(expected)

    def _write_manifest(self, manifest: Mapping[str, Any]) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.manifest_path.parent,
                prefix=f".{self.manifest_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as output:
                temporary_path = Path(output.name)
                json.dump(manifest, output, indent=2, sort_keys=True)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_path, self.manifest_path)
        except OSError as exc:
            if temporary_path is not None:
                with suppress(OSError):
                    temporary_path.unlink(missing_ok=True)
            raise DatabaseError(f"Failed to persist index manifest: {exc}") from exc

    def _validate_vector_map(self, vectors: Mapping[str, Sequence[float]]) -> None:
        if set(vectors) != set(self.VECTOR_SIZES):
            raise DatabaseError(
                f"Point vector names {sorted(vectors)} do not match {sorted(self.VECTOR_SIZES)}."
            )
        for name, size in self.VECTOR_SIZES.items():
            validate_vector(vectors[name], size, name)

    @staticmethod
    def _models() -> Any:
        try:
            from qdrant_client import models

            return models
        except ImportError as exc:
            raise DatabaseError(
                "qdrant-client is not installed. Run `pip install -r requirements.txt`."
            ) from exc


def _distance_name(distance: Any) -> str:
    value = getattr(distance, "value", distance)
    normalized = str(value).lower()
    return normalized.rsplit(".", maxsplit=1)[-1]


def _normalize_json(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(dict(value), sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise DatabaseError(f"Index manifest is not JSON serializable: {exc}") from exc
