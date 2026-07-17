"""Strict query-routing and retrieval-result data contracts."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class QueryPlan(BaseModel):
    """Three routed text queries and exact decimal weights returned by the LLM."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    garment_query: str = Field(
        min_length=1,
        max_length=500,
        description="Expanded search text describing only garments, materials, fit, and details.",
    )
    region_query: str = Field(
        min_length=1,
        max_length=500,
        description="Expanded search text describing the physical scene, setting, and location.",
    )
    vibe_query: str = Field(
        min_length=1,
        max_length=500,
        description="Expanded search text describing mood, aesthetic, era, and styling subculture.",
    )
    alpha: float = Field(ge=0.0, le=1.0, description="Garment importance weight.")
    beta: float = Field(ge=0.0, le=1.0, description="Region importance weight.")
    gamma: float = Field(ge=0.0, le=1.0, description="Vibe importance weight.")

    @model_validator(mode="after")
    def validate_exact_weight_sum(self) -> QueryPlan:
        total = sum(
            (Decimal(str(weight)) for weight in (self.alpha, self.beta, self.gamma)),
            start=Decimal("0"),
        )
        if total != Decimal("1"):
            raise ValueError(
                "alpha + beta + gamma must equal exactly 1.0 in decimal notation; "
                f"received {total}."
            )
        return self


class BranchScores(BaseModel):
    """Raw cosine scores used in the exact weighted fusion formula."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    garment: float
    region: float
    vibe: float


class RetrievalHit(BaseModel):
    """One fused and ranked image candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    image_path: str | None
    final_score: float
    branch_scores: BranchScores
    payload: dict[str, Any]


class RetrievalResponse(BaseModel):
    """Serializable end-to-end result returned by the retrieval CLI."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_query: str
    routing: QueryPlan
    candidate_counts: dict[str, int]
    missing_branch_score: float
    results: list[RetrievalHit]
