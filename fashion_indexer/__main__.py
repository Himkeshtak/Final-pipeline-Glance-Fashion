"""Allow ``python -m fashion_indexer`` execution."""

from __future__ import annotations

from fashion_indexer.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
