"""Allow ``python -m fashion_retriever`` execution."""

from __future__ import annotations

from fashion_retriever.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
