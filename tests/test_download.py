"""The NIH downloader plans archives and unpacks local files without a network call."""

from __future__ import annotations

import gzip
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from cxr_triage.data.download import (
    NIH_IMAGE_ARCHIVES,
    download_file,
    extract_pngs,
    main,
    planned_archives,
    run_download,
)
from cxr_triage.labels import NIH_BOX_PAGE


def test_plan_lists_twelve_unique_nih_archives():
    archives = planned_archives()
    assert len(archives) == 12
    assert [archive.filename for archive in archives] == [
        f"images_{number:03d}.tar.gz" for number in range(1, 13)
    ]
    assert len({archive.url for archive in archives}) == 12
    assert archives[0].url == NIH_IMAGE_ARCHIVES[0][1]
    assert archives[-1].filename == "images_012.tar.gz"
    assert planned_archives([1, 12])[1].filename == "images_012.tar.gz"
    with pytest.raises(ValueError):
        planned_archives([13])


def test_dry_run_does_not_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog):
    def explode(*args, **kwargs):
        raise AssertionError("dry-run tried to use the network")

    monkeypatch.setattr(urllib.request, "urlopen", explode)
    caplog.set_level("INFO")
    run_download(tmp_path, dry_run=True)
    assert "images_001.tar.gz" in caplog.text
    assert NIH_BOX_PAGE in caplog.text
    assert list(tmp_path.rglob("*")) == []


def test_main_dry_run(tmp_path: Path, caplog):
    caplog.set_level("INFO")
    main(["--output-dir", str(tmp_path), "--dry-run", "--only", "1,2"])
    assert "images_001.tar.gz" in caplog.text
    assert "images_002.tar.gz" in caplog.text
    assert "images_003.tar.gz" not in caplog.text


def test_download_file_checks_gzip_magic(tmp_path: Path):
    bad = tmp_path / "bad.gz"
    bad.write_bytes(b"this is not gzip")
    with pytest.raises(RuntimeError, match="not a gzip"):
        download_file(bad.as_uri(), tmp_path / "out" / "images_001.tar.gz", expect_gzip=True)

    good = tmp_path / "good.gz"
    payload = gzip.compress(b"archive-bytes")
    good.write_bytes(payload)
    dest = tmp_path / "nested" / "images_001.tar.gz"
    download_file(good.as_uri(), dest, expect_gzip=True)
    assert dest.read_bytes() == payload
    download_file("file:///does-not-matter", dest, expect_gzip=True)
    assert dest.read_bytes() == payload


def test_extract_pngs_flattens_names_and_stays_inside_the_output_dir(tmp_path: Path):
    source = tmp_path / "00000001_000.png"
    Image.new("L", (8, 8), color=20).save(source)
    archive = tmp_path / "images_001.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(source, arcname="images/nested/00000001_000.png")
        handle.add(source, arcname="../escape.png")
    destination = tmp_path / "images"
    assert extract_pngs(archive, destination) == 2
    assert (destination / "00000001_000.png").is_file()
    assert (destination / "escape.png").is_file()
    assert not (tmp_path / "escape.png").is_file()


def test_from_dir_unpacks_zip_and_copies_loose_pngs(tmp_path: Path):
    loose = tmp_path / "local" / "b.png"
    loose.parent.mkdir()
    Image.new("L", (4, 4), color=3).save(loose)
    packed = tmp_path / "local" / "pack.zip"
    member = tmp_path / "a.png"
    Image.new("L", (4, 4), color=9).save(member)
    with zipfile.ZipFile(packed, "w") as handle:
        handle.write(member, arcname="folder/a.png")
    output = tmp_path / "data"
    run_download(output, from_dir=tmp_path / "local")
    assert (output / "images" / "a.png").is_file()
    assert (output / "images" / "b.png").is_file()
