from __future__ import annotations

import json
from types import SimpleNamespace

from fashion_retriever.config import LLMSettings
from fashion_retriever.llm import OpenRouterQueryParser


class FakeCompletions:
    def __init__(self, contents: list[str]) -> None:
        self.contents = iter(contents)
        self.requests: list[dict] = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=next(self.contents)))]
        )


def _fake_client(contents: list[str]):
    completions = FakeCompletions(contents)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


def test_parser_requests_strict_schema_and_returns_valid_plan() -> None:
    content = json.dumps(
        {
            "garment_query": "oversized black leather jacket",
            "region_query": "rainy train station platform at night",
            "vibe_query": "vintage cinematic urban melancholy",
            "alpha": 0.5,
            "beta": 0.3,
            "gamma": 0.2,
        }
    )
    client, completions = _fake_client([content])
    parser = OpenRouterQueryParser("", LLMSettings(), client=client)

    plan = parser.parse("Vintage leather jacket at a rainy station")

    assert plan.garment_query == "oversized black leather jacket"
    request = completions.requests[0]
    assert request["response_format"]["type"] == "json_schema"
    assert request["response_format"]["json_schema"]["strict"] is True
    assert request["extra_body"]["provider"]["require_parameters"] is True


def test_parser_retries_a_weight_validation_failure() -> None:
    invalid = json.dumps(
        {
            "garment_query": "jacket",
            "region_query": "station",
            "vibe_query": "vintage",
            "alpha": 0.3,
            "beta": 0.3,
            "gamma": 0.3,
        }
    )
    valid = json.dumps(
        {
            "garment_query": "jacket",
            "region_query": "station",
            "vibe_query": "vintage",
            "alpha": 0.4,
            "beta": 0.3,
            "gamma": 0.3,
        }
    )
    client, completions = _fake_client([invalid, valid])
    parser = OpenRouterQueryParser("", LLMSettings(validation_attempts=2), client=client)

    plan = parser.parse("jacket at a station")

    assert plan.alpha == 0.4
    assert len(completions.requests) == 2
    assert "failed validation" in completions.requests[1]["messages"][-1]["content"]
