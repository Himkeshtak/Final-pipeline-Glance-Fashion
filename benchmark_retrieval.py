"""Measure and report an end-to-end multi-expert fashion retrieval query."""

from __future__ import annotations

import argparse
import html
import json
import logging
import os
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from fashion_indexer.config import COLLECTION_NAME, ModelSettings
from fashion_indexer.device import resolve_device
from fashion_indexer.environment import load_environment
from fashion_indexer.errors import FashionIndexerError
from fashion_retriever.config import DEFAULT_OPENROUTER_MODEL, DEFAULT_QUERY, LLMSettings
from fashion_retriever.database import FashionSearchStore
from fashion_retriever.encoders import MultiExpertTextEncoder
from fashion_retriever.errors import RetrieverError
from fashion_retriever.fusion import fuse_results
from fashion_retriever.llm import OpenRouterQueryParser

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark one routed query against the multi-expert fashion index."
    )
    parser.add_argument("query", nargs="?", default=DEFAULT_QUERY)
    parser.add_argument("--db-path", type=Path, default=Path("qdrant_fashion_db"))
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--candidate-limit", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--missing-score", type=float, default=0.0)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-timeout", type=float, default=45.0)
    parser.add_argument("--model-cache-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument(
        "--precision",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--sequential-text-encoding", action="store_true")
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-html", type=Path, default=None)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    load_environment(args.env_file)
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RetrieverError("OPENROUTER_API_KEY is not configured.")
    if args.candidate_limit < 1 or args.top_k < 1:
        raise RetrieverError("candidate-limit and top-k must both be at least 1.")

    device_spec = resolve_device(args.device, args.precision)
    model_settings = ModelSettings(cache_dir=args.model_cache_dir)
    llm_settings = LLMSettings(
        model=args.llm_model
        or os.environ.get("OPENROUTER_MODEL", "").strip()
        or DEFAULT_OPENROUTER_MODEL,
        timeout_seconds=args.llm_timeout,
    )

    benchmark_started = perf_counter()
    with FashionSearchStore(args.db_path, args.collection) as store:
        indexed_points = store.validate_ready(model_settings)

        model_load_started = perf_counter()
        parser = OpenRouterQueryParser(api_key, llm_settings)
        encoder = MultiExpertTextEncoder(
            model_settings,
            device_spec,
            parallel=not args.sequential_text_encoding,
        )
        model_load_seconds = perf_counter() - model_load_started

        routing_started = perf_counter()
        plan = parser.parse(args.query)
        routing_seconds = perf_counter() - routing_started

        encoding_started = perf_counter()
        vectors = encoder.encode(plan)
        text_encoding_seconds = perf_counter() - encoding_started

        search_started = perf_counter()
        branch_results = store.search_all(vectors, limit=args.candidate_limit)
        qdrant_search_seconds = perf_counter() - search_started

        fusion_started = perf_counter()
        results = fuse_results(
            branch_results,
            plan,
            missing_score=args.missing_score,
            top_k=args.top_k,
        )
        fusion_seconds = perf_counter() - fusion_started

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": args.query,
        "device": device_spec.description,
        "indexed_points": indexed_points,
        "candidate_limit_per_branch": args.candidate_limit,
        "top_k": args.top_k,
        "routing": plan.model_dump(),
        "candidate_counts": {name: len(hits) for name, hits in branch_results.items()},
        "timings_seconds": {
            "model_loading": model_load_seconds,
            "llm_routing": routing_seconds,
            "parallel_text_encoding": text_encoding_seconds,
            "qdrant_batch_search": qdrant_search_seconds,
            "score_fusion": fusion_seconds,
            "warm_query_total": (
                routing_seconds + text_encoding_seconds + qdrant_search_seconds + fusion_seconds
            ),
            "process_total": perf_counter() - benchmark_started,
        },
        "results": [
            {"rank": rank, **result.model_dump()} for rank, result in enumerate(results, start=1)
        ],
    }
    return report


def write_html_report(report: dict[str, Any], output_path: Path) -> Path:
    """Write a self-contained benchmark gallery that references the source images."""
    routing = report["routing"]
    timings = report["timings_seconds"]
    cards: list[str] = []
    for result in report["results"]:
        image_path = Path(result["image_path"]).expanduser().resolve()
        image_uri = image_path.as_uri()
        branch_scores = result["branch_scores"]
        cards.append(
            f"""
            <article class="card">
              <img src="{html.escape(image_uri, quote=True)}"
                   alt="Retrieval result rank {int(result["rank"])}" loading="lazy">
              <div class="card-body">
                <h2>#{int(result["rank"])} · {float(result["final_score"]):.4f}</h2>
                <dl>
                  <div><dt>Garment</dt><dd>{float(branch_scores["garment"]):.4f}</dd></div>
                  <div><dt>Region</dt><dd>{float(branch_scores["region"]):.4f}</dd></div>
                  <div><dt>Vibe</dt><dd>{float(branch_scores["vibe"]):.4f}</dd></div>
                </dl>
                <a href="{html.escape(image_uri, quote=True)}">{html.escape(image_path.name)}</a>
              </div>
            </article>
            """
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fashion retrieval benchmark</title>
  <style>
    :root {{ color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
    body {{ margin: 0; background: #0b0d12; color: #eef1f7; }}
    main {{ max-width: 1440px; margin: auto; padding: 32px; }}
    h1 {{ margin-bottom: 8px; }}
    .muted {{ color: #aab2c3; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      gap: 12px; margin: 24px 0; }}
    .summary section {{ background: #151925; border: 1px solid #252b3b; border-radius: 12px;
      padding: 16px; }}
    .summary h2 {{ margin: 0 0 8px; font-size: 15px; color: #8eb8ff; }}
    .summary p {{ margin: 0; line-height: 1.45; }}
    .gallery {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 18px; }}
    .card {{ overflow: hidden; background: #151925; border: 1px solid #252b3b;
      border-radius: 14px; }}
    .card img {{ width: 100%; height: 390px; display: block; object-fit: cover;
      background: #090b0f; }}
    .card-body {{ padding: 14px; }}
    .card h2 {{ margin: 0 0 10px; font-size: 20px; }}
    dl {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin: 0 0 12px; }}
    dl div {{ background: #0f121a; border-radius: 8px; padding: 8px; }}
    dt {{ color: #9ca6b9; font-size: 12px; }}
    dd {{ margin: 2px 0 0; font-variant-numeric: tabular-nums; }}
    a {{ color: #8eb8ff; overflow-wrap: anywhere; }}
  </style>
</head>
<body><main>
  <h1>Multi-expert fashion retrieval benchmark</h1>
  <p class="muted">{html.escape(str(report["query"]))}</p>
  <div class="summary">
    <section><h2>Routing weights</h2><p>Garment {float(routing["alpha"]):.2f} ·
      Region {float(routing["beta"]):.2f} · Vibe {float(routing["gamma"]):.2f}</p></section>
    <section><h2>Garment query</h2><p>{html.escape(str(routing["garment_query"]))}</p></section>
    <section><h2>Region query</h2><p>{html.escape(str(routing["region_query"]))}</p></section>
    <section><h2>Vibe query</h2><p>{html.escape(str(routing["vibe_query"]))}</p></section>
    <section><h2>Runtime</h2><p>Warm query {float(timings["warm_query_total"]):.3f}s ·
      Cold model load {float(timings["model_loading"]):.3f}s ·
      {int(report["indexed_points"])} indexed images</p></section>
  </div>
  <section class="gallery">{"".join(cards)}</section>
</main></body>
</html>
"""
    resolved_path = output_path.expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(document, encoding="utf-8")
    return resolved_path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stderr,
    )
    try:
        report = run_benchmark(args)
        serialized = json.dumps(report, indent=2, ensure_ascii=False)
        print(serialized)
        if args.output_json is not None:
            output_path = args.output_json.expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(serialized + "\n", encoding="utf-8")
            LOGGER.info("Wrote benchmark report to %s", output_path)
        if args.output_html is not None:
            html_path = write_html_report(report, args.output_html)
            LOGGER.info("Wrote benchmark gallery to %s", html_path)
        return 0
    except (RetrieverError, FashionIndexerError) as exc:
        LOGGER.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        LOGGER.warning("Benchmark interrupted by user.")
        return 130
    except Exception as exc:
        LOGGER.exception("Unexpected benchmark failure: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
