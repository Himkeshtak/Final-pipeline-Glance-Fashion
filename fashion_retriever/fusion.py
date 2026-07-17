"""Exact weighted linear fusion for three named-vector result lists."""

from __future__ import annotations

from typing import Any

from fashion_indexer.config import (
    GARMENT_VECTOR_NAME,
    REGION_VECTOR_NAME,
    VIBE_VECTOR_NAME,
)
from fashion_retriever.database import BranchHit
from fashion_retriever.schemas import BranchScores, QueryPlan, RetrievalHit


def fuse_results(
    branch_results: dict[str, list[BranchHit]],
    plan: QueryPlan,
    *,
    missing_score: float,
    top_k: int,
) -> list[RetrievalHit]:
    """Fuse the union of candidates using alpha*g + beta*r + gamma*v."""

    candidates: dict[str, dict[str, Any]] = {}
    for branch_name in (GARMENT_VECTOR_NAME, REGION_VECTOR_NAME, VIBE_VECTOR_NAME):
        for hit in branch_results.get(branch_name, []):
            candidate = candidates.setdefault(
                hit.id,
                {
                    "scores": {
                        GARMENT_VECTOR_NAME: missing_score,
                        REGION_VECTOR_NAME: missing_score,
                        VIBE_VECTOR_NAME: missing_score,
                    },
                    "payload": {},
                },
            )
            candidate["scores"][branch_name] = hit.score
            candidate["payload"].update(hit.payload)

    fused: list[RetrievalHit] = []
    for point_id, candidate in candidates.items():
        scores = candidate["scores"]
        final_score = (
            plan.alpha * scores[GARMENT_VECTOR_NAME]
            + plan.beta * scores[REGION_VECTOR_NAME]
            + plan.gamma * scores[VIBE_VECTOR_NAME]
        )
        payload = dict(candidate["payload"])
        image_path = payload.get("image_path")
        fused.append(
            RetrievalHit(
                id=point_id,
                image_path=image_path if isinstance(image_path, str) else None,
                final_score=final_score,
                branch_scores=BranchScores(
                    garment=scores[GARMENT_VECTOR_NAME],
                    region=scores[REGION_VECTOR_NAME],
                    vibe=scores[VIBE_VECTOR_NAME],
                ),
                payload=payload,
            )
        )

    fused.sort(key=lambda result: (-result.final_score, result.id))
    return fused[:top_k]
