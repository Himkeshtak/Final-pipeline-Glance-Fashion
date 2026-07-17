from __future__ import annotations

import math
import uuid
from pathlib import Path

import pytest

from fashion_indexer.config import COLLECTION_NAME
from fashion_indexer.database import FashionVectorStore
from fashion_indexer.errors import CollectionSchemaError
from fashion_indexer.experts import NamedEmbeddings

qdrant_client = pytest.importorskip("qdrant_client")
QdrantClient = qdrant_client.QdrantClient
models = qdrant_client.models


def _unit_vector(size: int, index: int = 0) -> list[float]:
    vector = [0.0] * size
    vector[index] = 1.0
    assert math.isclose(sum(value * value for value in vector), 1.0)
    return vector


def _embeddings() -> NamedEmbeddings:
    return NamedEmbeddings(
        garment_vector=_unit_vector(512),
        region_vector=_unit_vector(768, 1),
        vibe_vector=_unit_vector(768, 2),
    )


def test_named_collection_upsert_payload_uuid_and_count(tmp_path: Path) -> None:
    with FashionVectorStore(tmp_path / "db", COLLECTION_NAME) as store:
        store.ensure_collection()
        vectors = store.client.get_collection(COLLECTION_NAME).config.params.vectors
        assert set(vectors) == {"garment_vector", "region_vector", "vibe_vector"}
        assert vectors["garment_vector"].size == 512
        assert vectors["region_vector"].size == 768
        assert vectors["vibe_vector"].size == 768
        assert all(params.distance == models.Distance.COSINE for params in vectors.values())
        point = store.make_point(
            _embeddings(),
            image_path="/catalog/look.jpg",
            file_hash="a" * 32,
            timestamp="2026-01-02T03:04:05+00:00",
        )
        parsed_id = uuid.UUID(str(point.id))

        assert parsed_id.version == 4
        assert set(point.vector) == {"garment_vector", "region_vector", "vibe_vector"}
        assert point.payload == {
            "image_path": "/catalog/look.jpg",
            "file_hash": "a" * 32,
            "timestamp": "2026-01-02T03:04:05+00:00",
        }

        store.upsert([point])
        assert store.count() == 1
        assert store.existing_file_hashes() == {"a" * 32}


def test_incompatible_existing_schema_is_not_overwritten(tmp_path: Path) -> None:
    db_path = tmp_path / "wrong-db"
    client = QdrantClient(path=str(db_path))
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "garment_vector": models.VectorParams(size=7, distance=models.Distance.COSINE)
        },
    )
    store = FashionVectorStore(db_path, COLLECTION_NAME, client=client)
    try:
        with pytest.raises(CollectionSchemaError, match="expected exactly"):
            store.ensure_collection()
        info = client.get_collection(COLLECTION_NAME)
        assert info.config.params.vectors["garment_vector"].size == 7
    finally:
        store.close()


def test_explicit_nondefault_datatype_is_rejected(tmp_path: Path) -> None:
    db_path = tmp_path / "typed-db"
    client = QdrantClient(path=str(db_path))
    vector_config = {
        "garment_vector": models.VectorParams(
            size=512,
            distance=models.Distance.COSINE,
            datatype=models.Datatype.FLOAT16,
        ),
        "region_vector": models.VectorParams(size=768, distance=models.Distance.COSINE),
        "vibe_vector": models.VectorParams(size=768, distance=models.Distance.COSINE),
    }
    client.create_collection(collection_name=COLLECTION_NAME, vectors_config=vector_config)
    store = FashionVectorStore(db_path, COLLECTION_NAME, client=client)
    try:
        with pytest.raises(CollectionSchemaError, match="explicit datatype"):
            store.ensure_collection()
    finally:
        store.close()


def test_manifest_prevents_mixed_embedding_contracts(tmp_path: Path) -> None:
    db_path = tmp_path / "manifest-db"
    with FashionVectorStore(
        db_path,
        COLLECTION_NAME,
        index_manifest={"model_revision": "first"},
    ) as store:
        store.ensure_collection()
        store.upsert(
            [
                store.make_point(
                    _embeddings(),
                    image_path="/catalog/look.jpg",
                    file_hash="b" * 32,
                )
            ]
        )

    with (
        FashionVectorStore(
            db_path,
            COLLECTION_NAME,
            index_manifest={"model_revision": "second"},
        ) as store,
        pytest.raises(CollectionSchemaError, match="different model revisions"),
    ):
        store.ensure_collection()
