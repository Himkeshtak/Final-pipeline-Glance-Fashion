from __future__ import annotations

import os
from pathlib import Path

import pytest

from fashion_indexer.environment import load_environment
from fashion_indexer.errors import ConfigurationError


@pytest.mark.parametrize("source", ["HUGGING_FACE_API", "HUGGING_FACE_HUB_TOKEN"])
def test_load_environment_aliases_hugging_face_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(f"{source}=test-secret\n", encoding="utf-8")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_API", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    status = load_environment(dotenv_path)

    assert os.environ["HF_TOKEN"] == "test-secret"
    assert status.dotenv_path == dotenv_path.resolve()
    assert status.huggingface_token_configured is True
    assert status.huggingface_token_source == source


def test_load_environment_does_not_override_hf_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("HUGGING_FACE_API=file-secret\n", encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN", "process-secret")
    monkeypatch.delenv("HUGGING_FACE_API", raising=False)

    status = load_environment(dotenv_path)

    assert os.environ["HF_TOKEN"] == "process-secret"
    assert status.huggingface_token_source == "HF_TOKEN"


def test_explicit_missing_environment_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="Environment file does not exist"):
        load_environment(tmp_path / "missing.env")
