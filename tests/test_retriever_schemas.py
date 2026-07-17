from __future__ import annotations

import pytest
from pydantic import ValidationError

from fashion_retriever.schemas import QueryPlan


def test_query_plan_accepts_exact_decimal_weight_sum() -> None:
    plan = QueryPlan(
        garment_query="oversized leather jacket",
        region_query="rainy train station platform",
        vibe_query="vintage cinematic melancholy",
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
    )

    assert plan.alpha + plan.beta + plan.gamma == 1.0


def test_query_plan_rejects_inexact_weight_sum() -> None:
    with pytest.raises(ValidationError, match="must equal exactly 1.0"):
        QueryPlan(
            garment_query="jacket",
            region_query="station",
            vibe_query="vintage",
            alpha=0.33,
            beta=0.33,
            gamma=0.33,
        )


def test_query_plan_rejects_empty_branch_query() -> None:
    with pytest.raises(ValidationError):
        QueryPlan(
            garment_query=" ",
            region_query="station",
            vibe_query="vintage",
            alpha=0.5,
            beta=0.25,
            gamma=0.25,
        )
