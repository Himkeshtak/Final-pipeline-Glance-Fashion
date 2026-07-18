from __future__ import annotations

from pathlib import Path

from fashion_indexer.cli import main, select_random_subset


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


def test_random_subset_is_reproducible_and_without_replacement(tmp_path: Path) -> None:
    paths = [tmp_path / f"image-{index}.jpg" for index in range(20)]

    first = select_random_subset(paths, max_images=8, seed=2020)
    second = select_random_subset(paths, max_images=8, seed=2020)
    different = select_random_subset(paths, max_images=8, seed=2021)

    assert first == second
    assert first != different
    assert len(first) == 8
    assert len(set(first)) == 8
    assert set(first).issubset(paths)
