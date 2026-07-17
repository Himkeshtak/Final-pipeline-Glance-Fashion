from __future__ import annotations

import pytest

from fashion_indexer.config import (
    GARMENT_VECTOR_NAME,
    REGION_VECTOR_NAME,
    VIBE_VECTOR_NAME,
)
from fashion_retriever.database import BranchHit
from fashion_retriever.fusion import fuse_results
from fashion_retriever.schemas import QueryPlan


def test_fusion_uses_exact_weighted_formula_and_missing_baseline() -> None:
    plan = QueryPlan(
        garment_query="leather jacket",
        region_query="train station",
        vibe_query="vintage",
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
    )
    branches = {
        GARMENT_VECTOR_NAME: [
            BranchHit("a", 0.9, {"image_path": "a.jpg"}),
            BranchHit("b", 0.7, {"image_path": "b.jpg"}),
        ],
        REGION_VECTOR_NAME: [BranchHit("a", 0.4, {"file_hash": "hash-a"})],
        VIBE_VECTOR_NAME: [BranchHit("b", 0.8, {"file_hash": "hash-b"})],
    }

    results = fuse_results(branches, plan, missing_score=0.0, top_k=10)

    assert [result.id for result in results] == ["a", "b"]
    assert results[0].final_score == pytest.approx(0.5 * 0.9 + 0.3 * 0.4 + 0.2 * 0.0)
    assert results[1].final_score == pytest.approx(0.5 * 0.7 + 0.3 * 0.0 + 0.2 * 0.8)
    assert results[0].branch_scores.vibe == 0.0
    assert results[1].branch_scores.region == 0.0


def test_fusion_sorts_deterministically_and_applies_top_k() -> None:
    plan = QueryPlan(
        garment_query="coat",
        region_query="city",
        vibe_query="minimal",
        alpha=1.0,
        beta=0.0,
        gamma=0.0,
    )
    branches = {
        GARMENT_VECTOR_NAME: [
            BranchHit("b", 0.8, {}),
            BranchHit("a", 0.8, {}),
        ]
    }

    results = fuse_results(branches, plan, missing_score=0.0, top_k=1)

    assert [result.id for result in results] == ["a"]
