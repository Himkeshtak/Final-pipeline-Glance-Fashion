"""End-to-end orchestration from conversational query to fused image results."""

from __future__ import annotations

from typing import Protocol

from fashion_retriever.config import RetrieverSettings
from fashion_retriever.database import FashionSearchStore
from fashion_retriever.encoders import MultiExpertTextEncoder
from fashion_retriever.fusion import fuse_results
from fashion_retriever.schemas import QueryPlan, RetrievalResponse


class QueryParser(Protocol):
    def parse(self, raw_query: str) -> QueryPlan: ...


class FashionRetriever:
    """Coordinate LLM routing, text experts, Qdrant search, and score fusion."""

    def __init__(
        self,
        parser: QueryParser,
        encoder: MultiExpertTextEncoder,
        store: FashionSearchStore,
        settings: RetrieverSettings,
    ) -> None:
        self.parser = parser
        self.encoder = encoder
        self.store = store
        self.settings = settings

    def search(self, raw_query: str) -> RetrievalResponse:
        plan = self.parser.parse(raw_query)
        vectors = self.encoder.encode(plan)
        branch_results = self.store.search_all(
            vectors,
            limit=self.settings.candidate_limit,
        )
        results = fuse_results(
            branch_results,
            plan,
            missing_score=self.settings.missing_score,
            top_k=self.settings.top_k,
        )
        return RetrievalResponse(
            raw_query=raw_query.strip(),
            routing=plan,
            candidate_counts={name: len(hits) for name, hits in branch_results.items()},
            missing_branch_score=self.settings.missing_score,
            results=results,
        )
