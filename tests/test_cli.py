from __future__ import annotations

from pathlib import Path

from fashion_indexer.cli import main


def test_main_creates_and_verifies_collection_for_empty_directory(tmp_path: Path, capsys) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    exit_code = main(
        [
            "--images-dir",
            str(images_dir),
            "--db-path",
            str(tmp_path / "qdrant"),
            "--device",
            "cpu",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "collection='fashion_multi_expert_search', points=0" in captured.out


def test_main_reports_missing_image_directory(tmp_path: Path) -> None:
    exit_code = main(
        [
            "--images-dir",
            str(tmp_path / "missing"),
            "--db-path",
            str(tmp_path / "qdrant"),
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 2
