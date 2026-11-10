"""Download NIH ChestX-ray14 images from the Clinical Center release.

The 12 archive URLs are the batch links published with the Box folder
https://nihcc.app.box.com/v/ChestXray-NIHCC. The label CSV is a separate small
file in that same folder. This module does not vendor images or the CSV.

Nothing here talks to AWS or needs credentials. The download is large, about
45 GB, and is meant to be run by the project owner on a machine with disk to
spare.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from cxr_triage.labels import EXPECTED_IMAGE_COUNT, EXPECTED_PATIENT_COUNT, NIH_BOX_PAGE

logger = logging.getLogger(__name__)

# Published by the NIH Clinical Center alongside the ChestX-ray14 Box release.
# If a link starts returning HTML instead of a gzip archive, download that
# archive by hand from NIH_BOX_PAGE. Do not commit the archives.
NIH_IMAGE_ARCHIVES: tuple[tuple[str, str], ...] = (
    ("images_001.tar.gz", "https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz"),
    ("images_002.tar.gz", "https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz"),
    ("images_003.tar.gz", "https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz"),
    ("images_004.tar.gz", "https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz"),
    ("images_005.tar.gz", "https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz"),
    ("images_006.tar.gz", "https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz"),
    ("images_007.tar.gz", "https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz"),
    ("images_008.tar.gz", "https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz"),
    ("images_009.tar.gz", "https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz"),
    ("images_010.tar.gz", "https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz"),
    ("images_011.tar.gz", "https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz"),
    ("images_012.tar.gz", "https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz"),
)


@dataclass(frozen=True)
class Archive:
    filename: str
    url: str
    number: int


def planned_archives(only: list[int] | None = None) -> list[Archive]:
    """Return the NIH image archives, optionally filtered by 1-based index."""
    selected = []
    wanted = set(only) if only else None
    for number, (filename, url) in enumerate(NIH_IMAGE_ARCHIVES, start=1):
        if wanted is not None and number not in wanted:
            continue
        selected.append(Archive(filename=filename, url=url, number=number))
    if wanted is not None:
        known = {archive.number for archive in selected}
        missing = sorted(wanted.difference(known))
        if missing:
            raise ValueError(f"Unknown archive numbers {missing}. Use 1 through 12.")
    if not selected:
        raise ValueError("No archives selected.")
    return selected


def download_file(url: str, dest: Path, *, expect_gzip: bool = False) -> None:
    """Stream ``url`` to ``dest``. An existing non-empty file is left in place."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        if expect_gzip and not is_gzip(dest):
            raise RuntimeError(f"{dest} exists but is not a gzip archive. Delete it and download again.")
        logger.info("Keeping existing file %s", dest)
        return

    temporary = dest.with_suffix(dest.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "cxr-triage/0.1"})
    logger.info("Downloading %s", dest.name)
    progress = _Progress(dest.name)
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                progress.update(len(chunk))
        temporary.replace(dest)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

    if expect_gzip and not is_gzip(dest):
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"Download for {dest.name} was not a gzip archive. The NIH link may have "
            f"changed. Download it manually from {NIH_BOX_PAGE}."
        )


def extract_pngs(archive: Path, image_dir: Path) -> int:
    """Extract PNG members into ``image_dir`` using only the file basename.

    Basenames match ``Image Index``. Nested paths and ``..`` members cannot
    write outside ``image_dir``.
    """
    image_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for name, reader in _iter_png_members(archive):
        target = image_dir / name
        with reader, target.open("wb") as handle:
            shutil.copyfileobj(reader, handle)
        written += 1
    logger.info("Extracted %s PNGs from %s", written, archive.name)
    return written


def is_gzip(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(2) == b"\x1f\x8b"


def run_download(
    output_dir: str | Path,
    *,
    dry_run: bool = False,
    only: list[int] | None = None,
    from_dir: str | Path | None = None,
    metadata_url: str | None = None,
    keep_archives: bool = True,
) -> None:
    """Download or unpack images into ``<output>/images`` and look for the CSV."""
    root = Path(output_dir)
    image_dir = root / "images"
    archive_dir = root / "archives"
    archives = planned_archives(only)
    if dry_run:
        logger.info("Dry run. About 45 GB would be downloaded to %s.", root)
        for archive in archives:
            logger.info("%s  %s", archive.filename, archive.url)
        logger.info("Label CSV: download Data_Entry_2017.csv from %s into %s", NIH_BOX_PAGE, root)
        return

    root.mkdir(parents=True, exist_ok=True)
    if from_dir is None:
        for archive in archives:
            destination = archive_dir / archive.filename
            download_file(archive.url, destination, expect_gzip=True)
            extract_pngs(destination, image_dir)
            if not keep_archives:
                destination.unlink(missing_ok=True)
    else:
        _ingest_local(Path(from_dir), image_dir)

    if metadata_url:
        download_file(metadata_url, root / "Data_Entry_2017.csv", expect_gzip=False)
    _report_metadata(root)


def _ingest_local(source: Path, image_dir: Path) -> None:
    """Copy loose PNGs and unpack local tar.gz or zip archives."""
    if not source.exists():
        raise FileNotFoundError(f"Local data directory not found: {source}")
    archives: list[Path] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        name = path.name.lower()
        if name.endswith(".tar.gz") or name.endswith(".tgz") or name.endswith(".zip"):
            archives.append(path)
            continue
        if name.endswith(".png"):
            image_dir.mkdir(parents=True, exist_ok=True)
            target = image_dir / path.name
            if path.resolve() != target.resolve():
                shutil.copy2(path, target)
    for archive in archives:
        extract_pngs(archive, image_dir)


def _iter_png_members(archive: Path):
    name = archive.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as handle:
            for info in handle.infolist():
                if info.is_dir() or not info.filename.lower().endswith(".png"):
                    continue
                basename = Path(info.filename).name
                if not basename:
                    continue
                yield basename, handle.open(info)
        return
    with tarfile.open(archive, "r:*") as handle:
        for member in handle.getmembers():
            if not member.isfile() or not member.name.lower().endswith(".png"):
                continue
            basename = Path(member.name).name
            if not basename:
                continue
            reader = handle.extractfile(member)
            if reader is None:
                continue
            yield basename, reader


def _report_metadata(root: Path) -> None:
    csv_path = root / "Data_Entry_2017.csv"
    if not csv_path.is_file():
        logger.warning(
            "Data_Entry_2017.csv is not in %s. Download it from %s and place it there. "
            "The image archives do not include the label file.",
            root,
            NIH_BOX_PAGE,
        )
        return
    # Imported lazily so `--dry-run` does not need pandas until a CSV exists.
    from cxr_triage.data.metadata import load_metadata

    frame = load_metadata(csv_path)
    patients = int(frame["patient_id"].nunique())
    if len(frame) != EXPECTED_IMAGE_COUNT or patients != EXPECTED_PATIENT_COUNT:
        logger.warning(
            "CSV has %s images and %s patients. The full NIH release has %s images "
            "from %s patients. A subset is fine for debugging.",
            len(frame),
            patients,
            EXPECTED_IMAGE_COUNT,
            EXPECTED_PATIENT_COUNT,
        )
    else:
        logger.info(
            "CSV matches the full release: %s images, %s patients.",
            EXPECTED_IMAGE_COUNT,
            EXPECTED_PATIENT_COUNT,
        )


class _Progress:
    def __init__(self, name: str) -> None:
        self.name = name
        self.seen = 0
        self.next_mark = 500 * 1024 * 1024

    def update(self, count: int) -> None:
        self.seen += count
        if self.seen >= self.next_mark:
            logger.info("%s: downloaded %.1f GB", self.name, self.seen / 1e9)
            self.next_mark += 500 * 1024 * 1024


def _parse_only(value: str | None) -> list[int] | None:
    if not value:
        return None
    numbers = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit():
            raise argparse.ArgumentTypeError(f"Archive id {part!r} is not an integer.")
        numbers.append(int(part))
    return numbers or None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download NIH ChestX-ray14 images.")
    parser.add_argument("--output-dir", default="data", help="Destination. Defaults to ./data")
    parser.add_argument("--dry-run", action="store_true", help="List archives and exit.")
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated archive numbers to download, for example 1,2.",
    )
    parser.add_argument(
        "--from-dir",
        default=None,
        help="Unpack local tar.gz or zip archives and copy loose PNGs instead of downloading.",
    )
    parser.add_argument(
        "--metadata-url",
        default=None,
        help="Optional direct URL for Data_Entry_2017.csv. The Box folder is the source of truth.",
    )
    parser.add_argument(
        "--delete-archives",
        action="store_true",
        help="Delete each archive after it is extracted. Keeps disk use closer to one copy.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_download(
        args.output_dir,
        dry_run=args.dry_run,
        only=_parse_only(args.only),
        from_dir=args.from_dir,
        metadata_url=args.metadata_url,
        keep_archives=not args.delete_archives,
    )


if __name__ == "__main__":
    main()
