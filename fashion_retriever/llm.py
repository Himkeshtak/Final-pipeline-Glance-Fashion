"""OpenRouter-backed structured query parsing with strict validation."""

from __future__ import annotations

import json
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from pydantic import ValidationError

from fashion_retriever.config import LLMSettings
from fashion_retriever.errors import LLMServiceError, LLMTimeoutError, QueryPlanError
from fashion_retriever.schemas import QueryPlan

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

SYSTEM_PROMPT = """You are the routing expert for a multi-vector fashion image search engine.
Decompose each conversational request into three standalone retrieval texts:
1. garment_query: clothing category, silhouette, fit, material, color, construction, and details.
2. region_query: physical setting, architecture, weather, geography, and background context.
3. vibe_query: era, mood, lighting, styling subculture, visual aesthetic, and composition.

Expand terse phrases into precise visual search language without inventing contradictory details.
If a branch is absent, return a neutral non-empty description for that branch and give it weight 0.
Choose alpha, beta, and gamma from the user's emphasis. They must be non-negative decimal numbers
and alpha + beta + gamma must equal exactly 1.0 in decimal arithmetic. Return only the schema.
"""


class OpenRouterQueryParser:
    """Convert a raw fashion request into a validated three-expert query plan."""

    def __init__(
        self,
        api_key: str,
        settings: LLMSettings,
        *,
        client: Any | None = None,
    ) -> None:
        if not api_key.strip() and client is None:
            raise LLMServiceError("OPENROUTER_API_KEY is not configured.")
        self.settings = settings
        self.client = client or OpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            timeout=settings.timeout_seconds,
            max_retries=settings.transport_retries,
        )

    def parse(self, raw_query: str) -> QueryPlan:
        query = raw_query.strip()
        if not query:
            raise QueryPlanError("The search query cannot be empty.")

        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        last_validation_error: Exception | None = None

        for attempt in range(1, self.settings.validation_attempts + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.settings.model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=self.settings.max_tokens,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "fashion_query_plan",
                            "strict": True,
                            "schema": QueryPlan.model_json_schema(),
                        },
                    },
                    extra_body={"provider": {"require_parameters": True}},
                )
            except APITimeoutError as exc:
                raise LLMTimeoutError(
                    f"OpenRouter query parsing timed out after "
                    f"{self.settings.timeout_seconds:.1f} seconds."
                ) from exc
            except APIConnectionError as exc:
                raise LLMServiceError(f"Cannot connect to OpenRouter: {exc}") from exc
            except APIStatusError as exc:
                raise LLMServiceError(
                    f"OpenRouter returned HTTP {exc.status_code}: {exc.message}"
                ) from exc
            except Exception as exc:
                raise LLMServiceError(f"OpenRouter query parsing failed: {exc}") from exc

            content = self._extract_content(response)
            try:
                plan = QueryPlan.model_validate(json.loads(content))
                return plan
            except (json.JSONDecodeError, ValidationError) as exc:
                last_validation_error = exc
                if attempt == self.settings.validation_attempts:
                    break
                messages.extend(
                    [
                        {"role": "assistant", "content": content},
                        {
                            "role": "user",
                            "content": (
                                "The prior object failed validation. Return a corrected object "
                                "with all required fields and alpha + beta + gamma exactly equal "
                                "to 1.0. "
                                f"Validation error: {exc}"
                            ),
                        },
                    ]
                )

        raise QueryPlanError(
            f"OpenRouter did not return a valid fashion query plan: {last_validation_error}"
        ) from last_validation_error

    @staticmethod
    def _extract_content(response: Any) -> str:
        choices = getattr(response, "choices", None)
        if not choices:
            raise LLMServiceError("OpenRouter returned no completion choices.")
        message = getattr(choices[0], "message", None)
        refusal = getattr(message, "refusal", None)
        if refusal:
            raise LLMServiceError(f"The routing model refused the query: {refusal}")
        content = getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise LLMServiceError("OpenRouter returned an empty structured response.")
        return content
