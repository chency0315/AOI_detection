from __future__ import annotations

from aoi_detection.data import iter_images


def test_finds_images_recursively_and_ignores_other_files(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.png").touch()
    (tmp_path / "nested" / "b.JPG").touch()
    (tmp_path / "notes.txt").touch()

    found = [p.name for p in iter_images(tmp_path)]
    assert found == ["a.png", "b.JPG"]


def test_single_file_input(tmp_path):
    image = tmp_path / "only.bmp"
    image.touch()
    assert list(iter_images(image)) == [image]


def test_non_recursive_skips_subdirectories(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "top.png").touch()
    (tmp_path / "nested" / "deep.png").touch()

    assert [p.name for p in iter_images(tmp_path, recursive=False)] == ["top.png"]
