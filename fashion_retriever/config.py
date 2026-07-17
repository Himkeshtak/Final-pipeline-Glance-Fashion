"""Validated settings for LLM routing, text encoding, and retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fashion_indexer.config import COLLECTION_NAME
from fashion_retriever.errors import RetrieverConfigurationError

DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-mini"
DEFAULT_QUERY = "Vintage oversized leather jacket at a rainy train station"


@dataclass(frozen=True, slots=True)
class LLMSettings:
    """OpenRouter structured-output request settings."""

    model: str = DEFAULT_OPENROUTER_MODEL
    timeout_seconds: float = 45.0
    transport_retries: int = 2
    validation_attempts: int = 2
    max_tokens: int = 600

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise RetrieverConfigurationError("The OpenRouter model identifier cannot be empty.")
        if self.timeout_seconds <= 0:
            raise RetrieverConfigurationError("LLM timeout must be greater than zero.")
        if self.transport_retries < 0:
            raise RetrieverConfigurationError("LLM transport retries cannot be negative.")
        if self.validation_attempts < 1:
            raise RetrieverConfigurationError("LLM validation attempts must be at least one.")
        if self.max_tokens < 100:
            raise RetrieverConfigurationError("LLM max_tokens must be at least 100.")


@dataclass(frozen=True, slots=True)
class RetrieverSettings:
    """Local Qdrant and score-fusion settings."""

    db_path: Path = Path("qdrant_fashion_db")
    collection_name: str = COLLECTION_NAME
    candidate_limit: int = 100
    top_k: int = 10
    missing_score: float = 0.0
    query_attempts: int = 3

    def __post_init__(self) -> None:
        if not self.collection_name.strip():
            raise RetrieverConfigurationError("Qdrant collection name cannot be empty.")
        if self.candidate_limit < 1:
            raise RetrieverConfigurationError("candidate_limit must be at least one.")
        if self.top_k < 1:
            raise RetrieverConfigurationError("top_k must be at least one.")
        if not -1.0 <= self.missing_score <= 1.0:
            raise RetrieverConfigurationError("missing_score must be between -1 and 1.")
        if self.query_attempts < 1:
            raise RetrieverConfigurationError("query_attempts must be at least one.")
