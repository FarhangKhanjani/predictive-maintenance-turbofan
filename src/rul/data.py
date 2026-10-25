"""Download, verify, and load the C-MAPSS dataset.

Reproducibility rests on two checks:
1. the downloaded zip must match a pinned SHA-256 (``config.DATASET_ZIP_SHA256``);
2. every extracted file must match the committed manifest ``data/SHA256SUMS``.

Command line:
    uv run rul-data download   # fetch and verify into data/raw
    uv run rul-data verify     # re-check files already on disk
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from rul import config


class ChecksumError(RuntimeError):
    """Raised when data on disk or from the network does not match the pinned checksum."""


def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 digest of a byte string."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path = config.CHECKSUM_FILE) -> dict[str, str]:
    """Parse a ``sha256sum``-style manifest into {file name: digest}."""
    manifest: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            manifest[name.lstrip("*")] = digest
    return manifest


def extract_members(archive: zipfile.ZipFile, wanted: set[str], target: Path) -> list[str]:
    """Extract files whose base name is in ``wanted`` into ``target``.

    The NASA archive wraps the data in an inner zip, so nested zips are
    opened recursively. Only the bare file name is used when writing, which
    also prevents path traversal from crafted archive entries.

    Returns:
        The names of the files that were written.
    """
    written: list[str] = []
    for member in archive.namelist():
        base = Path(member).name
        if base.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(archive.read(member))) as inner:
                written += extract_members(inner, wanted, target)
        elif base in wanted:
            (target / base).write_bytes(archive.read(member))
            written.append(base)
    return written


def verify(raw_dir: Path = config.RAW_DIR, manifest_path: Path = config.CHECKSUM_FILE) -> None:
    """Check every file listed in the manifest against its digest.

    Raises:
        ChecksumError: if a file is missing or its digest differs.
    """
    problems = []
    for name, expected in read_manifest(manifest_path).items():
        path = raw_dir / name
        if not path.exists():
            problems.append(f"missing: {name}")
        elif sha256_file(path) != expected:
            problems.append(f"checksum mismatch: {name}")
    if problems:
        raise ChecksumError("; ".join(problems))


def download(
    raw_dir: Path = config.RAW_DIR,
    url: str = config.DATASET_URL,
    expected_sha256: str = config.DATASET_ZIP_SHA256,
    manifest_path: Path = config.CHECKSUM_FILE,
) -> None:
    """Download the dataset zip, verify it, extract it, and verify each file.

    Skips the network entirely if all files are already present and valid.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        verify(raw_dir, manifest_path)
        print(f"Data already present and verified in {raw_dir}")
        return
    except ChecksumError:
        pass

    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = response.read()
    if sha256_bytes(payload) != expected_sha256:
        raise ChecksumError(f"downloaded zip does not match pinned SHA-256 {expected_sha256}")

    wanted = set(read_manifest(manifest_path))
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        extract_members(archive, wanted, raw_dir)
    verify(raw_dir, manifest_path)
    print(f"Extracted and verified {len(wanted)} files in {raw_dir}")


def load_split(subset: str, split: str, raw_dir: Path = config.RAW_DIR) -> pd.DataFrame:
    """Load ``train`` or ``test`` data for one subset with named columns.

    Args:
        subset: One of ``config.SUBSETS``, for example ``"FD001"``.
        split: ``"train"`` or ``"test"``.
        raw_dir: Directory containing the extracted text files.

    Returns:
        One row per (unit, cycle), columns ``config.COLUMNS``.
    """
    if subset not in config.SUBSETS:
        raise ValueError(f"unknown subset {subset!r}, expected one of {config.SUBSETS}")
    if split not in config.SPLITS:
        raise ValueError(f"unknown split {split!r}, expected one of {config.SPLITS}")
    df = pd.read_csv(raw_dir / f"{split}_{subset}.txt", sep=r"\s+", header=None)
    if df.shape[1] != len(config.COLUMNS):
        raise ValueError(f"expected {len(config.COLUMNS)} columns, found {df.shape[1]}")
    df.columns = config.COLUMNS
    return df.astype({"unit": "int64", "cycle": "int64"})


def load_test_rul(subset: str, raw_dir: Path = config.RAW_DIR) -> pd.Series:
    """Load the true RUL at the last cycle of each test engine, indexed by unit (1-based)."""
    rul = pd.read_csv(raw_dir / f"RUL_{subset}.txt", header=None).iloc[:, 0].astype("int64")
    rul.index = pd.RangeIndex(1, len(rul) + 1, name="unit")
    return rul.rename("rul")


def data_checksum(subset: str, raw_dir: Path = config.RAW_DIR) -> str:
    """Return one combined digest for a subset's train, test, and RUL files.

    Logged with every experiment so a run can be traced to the exact data.
    """
    names = [f"train_{subset}.txt", f"test_{subset}.txt", f"RUL_{subset}.txt"]
    combined = "".join(sha256_file(raw_dir / name) for name in names)
    return sha256_bytes(combined.encode())


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``uv run rul-data``."""
    parser = argparse.ArgumentParser(description="C-MAPSS data utilities")
    parser.add_argument("command", choices=["download", "verify"])
    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            download()
        else:
            verify()
            print(f"All files in {config.RAW_DIR} match {config.CHECKSUM_FILE.name}")
    except ChecksumError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
