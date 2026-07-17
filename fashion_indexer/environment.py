"""Safe loading and normalization of local runtime environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from fashion_indexer.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class EnvironmentStatus:
    """Non-secret environment loading status suitable for diagnostic logging."""

    dotenv_path: Path | None
    huggingface_token_configured: bool
    huggingface_token_source: str | None


def load_environment(dotenv_path: Path | None = None) -> EnvironmentStatus:
    """Load a dotenv file without overriding process variables and expose HF_TOKEN.

    ``HUGGING_FACE_API`` is accepted as a compatibility alias because some projects
    use that name for a Hugging Face access token. The token value is never returned
    or logged.
    """

    resolved_path = _resolve_dotenv_path(dotenv_path)
    if resolved_path is not None:
        load_dotenv(dotenv_path=resolved_path, override=False)

    token_source: str | None = None
    if os.environ.get("HF_TOKEN"):
        token_source = "HF_TOKEN"
    elif os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"]
        token_source = "HUGGING_FACE_HUB_TOKEN"
    elif os.environ.get("HUGGING_FACE_API"):
        os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_API"]
        token_source = "HUGGING_FACE_API"

    return EnvironmentStatus(
        dotenv_path=resolved_path,
        huggingface_token_configured=token_source is not None,
        huggingface_token_source=token_source,
    )


def _resolve_dotenv_path(dotenv_path: Path | None) -> Path | None:
    if dotenv_path is not None:
        resolved = dotenv_path.expanduser().resolve()
        if not resolved.is_file():
            raise ConfigurationError(f"Environment file does not exist: {resolved}")
        return resolved

    discovered = find_dotenv(usecwd=True)
    return Path(discovered).resolve() if discovered else None
