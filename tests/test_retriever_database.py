from __future__ import annotations

import math
from types import SimpleNamespace

from qdrant_client import QdrantClient, models

from fashion_retriever.database import FashionSearchStore
from fashion_retriever.encoders import EncodedQueries


def _unit_vector(size: int) -> list[float]:
    return [1.0] + [0.0] * (size - 1)


class FakeSearchClient:
    def __init__(self) -> None:
        self.requests = None

    def query_batch_points(self, *, collection_name, requests):
        self.requests = requests
        return [
            SimpleNamespace(
                points=[
                    SimpleNamespace(
                        id=f"point-{index}",
                        score=0.9 - index * 0.1,
                        payload={"image_path": f"image-{index}.jpg"},
                    )
                ]
            )
            for index in range(3)
        ]


def test_search_all_uses_one_named_vector_batch() -> None:
    client = FakeSearchClient()
    store = FashionSearchStore(
        path="unused",
        collection_name="fashion_multi_expert_search",
        client=client,
    )
    vectors = EncodedQueries(
        garment_vector=_unit_vector(512),
        region_vector=_unit_vector(768),
        vibe_vector=_unit_vector(768),
    )

    results = store.search_all(vectors, limit=100)

    assert list(results) == ["garment_vector", "region_vector", "vibe_vector"]
    assert [request.using for request in client.requests] == list(results)
    assert all(request.limit == 100 for request in client.requests)
    assert all(math.isfinite(hits[0].score) for hits in results.values())


def test_search_all_against_real_in_memory_named_vector_collection() -> None:
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="fashion_multi_expert_search",
        vectors_config={
            "garment_vector": models.VectorParams(size=512, distance=models.Distance.COSINE),
            "region_vector": models.VectorParams(size=768, distance=models.Distance.COSINE),
            "vibe_vector": models.VectorParams(size=768, distance=models.Distance.COSINE),
        },
    )
    client.upsert(
        collection_name="fashion_multi_expert_search",
        points=[
            models.PointStruct(
                id=1,
                vector={
                    "garment_vector": _unit_vector(512),
                    "region_vector": _unit_vector(768),
                    "vibe_vector": _unit_vector(768),
                },
                payload={"image_path": "matched.jpg"},
            )
        ],
        wait=True,
    )
    store = FashionSearchStore(
        path="unused",
        collection_name="fashion_multi_expert_search",
        client=client,
    )
    vectors = EncodedQueries(
        garment_vector=_unit_vector(512),
        region_vector=_unit_vector(768),
        vibe_vector=_unit_vector(768),
    )

    results = store.search_all(vectors, limit=100)

    assert all(hits[0].id == "1" for hits in results.values())
    assert all(hits[0].score == 1.0 for hits in results.values())
    assert all(hits[0].payload["image_path"] == "matched.jpg" for hits in results.values())
    client.close()
