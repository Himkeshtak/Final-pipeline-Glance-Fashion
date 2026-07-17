from __future__ import annotations

from fashion_indexer.config import (
    GARMENT_VECTOR_NAME,
    REGION_VECTOR_NAME,
    VIBE_VECTOR_NAME,
)
from fashion_retriever.config import RetrieverSettings
from fashion_retriever.database import BranchHit
from fashion_retriever.encoders import EncodedQueries
from fashion_retriever.pipeline import FashionRetriever
from fashion_retriever.schemas import QueryPlan


class FakeParser:
    def parse(self, raw_query: str) -> QueryPlan:
        return QueryPlan(
            garment_query="red dress",
            region_query="ballroom",
            vibe_query="romantic",
            alpha=0.6,
            beta=0.2,
            gamma=0.2,
        )


class FakeEncoder:
    def encode(self, plan: QueryPlan) -> EncodedQueries:
        return EncodedQueries(
            garment_vector=[1.0] + [0.0] * 511,
            region_vector=[1.0] + [0.0] * 767,
            vibe_vector=[1.0] + [0.0] * 767,
        )


class FakeStore:
    def search_all(self, vectors: EncodedQueries, *, limit: int):
        assert limit == 100
        return {
            GARMENT_VECTOR_NAME: [BranchHit("id-1", 0.8, {"image_path": "dress.jpg"})],
            REGION_VECTOR_NAME: [BranchHit("id-1", 0.7, {})],
            VIBE_VECTOR_NAME: [BranchHit("id-1", 0.9, {})],
        }


def test_pipeline_returns_serializable_routing_and_fused_result() -> None:
    retriever = FashionRetriever(
        FakeParser(),
        FakeEncoder(),
        FakeStore(),
        RetrieverSettings(),
    )

    response = retriever.search("Find a romantic red dress in a ballroom")

    assert response.routing.alpha == 0.6
    assert response.results[0].image_path == "dress.jpg"
    assert response.results[0].final_score == 0.8
    assert '"garment_query":"red dress"' in response.model_dump_json()
