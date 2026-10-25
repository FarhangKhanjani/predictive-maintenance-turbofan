"""Tests for rul.data: loading, checksum verification, and safe extraction."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from rul import config, data


def make_manifest(raw_dir: Path, names: list[str]) -> Path:
    """Write a SHA256SUMS manifest for the given files and return its path."""
    manifest = raw_dir.parent / "SHA256SUMS"
    lines = [f"{data.sha256_file(raw_dir / n)}  {n}" for n in names]
    manifest.write_text("\n".join(lines) + "\n")
    return manifest


def nested_zip(files: dict[str, bytes]) -> bytes:
    """Build a zip that contains an inner zip with ``files``, like the NASA archive."""
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    outer = io.BytesIO()
    with zipfile.ZipFile(outer, "w") as zf:
        zf.writestr("dataset/CMAPSSData.zip", inner.getvalue())
    return outer.getvalue()


class FakeResponse(io.BytesIO):
    """Minimal stand-in for the object returned by urllib.request.urlopen."""

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


# --- loading -----------------------------------------------------------------


def test_load_split_names_columns_and_types(raw_dir: Path) -> None:
    df = data.load_split("FD001", "train", raw_dir)
    assert list(df.columns) == config.COLUMNS
    assert df["unit"].dtype == "int64"
    assert df.groupby("unit")["cycle"].max().to_dict() == {1: 5, 2: 7}


def test_load_split_rejects_unknown_subset_and_split(raw_dir: Path) -> None:
    with pytest.raises(ValueError, match="subset"):
        data.load_split("FD009", "train", raw_dir)
    with pytest.raises(ValueError, match="split"):
        data.load_split("FD001", "validation", raw_dir)


def test_load_split_rejects_wrong_column_count(raw_dir: Path) -> None:
    (raw_dir / "train_FD001.txt").write_text("1 1 0.1 0.2\n")
    with pytest.raises(ValueError, match="columns"):
        data.load_split("FD001", "train", raw_dir)


def test_load_test_rul_is_indexed_by_unit(raw_dir: Path) -> None:
    rul = data.load_test_rul("FD001", raw_dir)
    assert rul.to_dict() == {1: 12, 2: 30}


# --- checksums -------------------------------------------------------------------


def test_verify_passes_for_untouched_files(raw_dir: Path) -> None:
    manifest = make_manifest(raw_dir, ["train_FD001.txt", "RUL_FD001.txt"])
    data.verify(raw_dir, manifest)


def test_verify_detects_tampered_and_missing_files(raw_dir: Path) -> None:
    manifest = make_manifest(raw_dir, ["train_FD001.txt", "RUL_FD001.txt"])
    (raw_dir / "RUL_FD001.txt").write_text("99\n")
    (raw_dir / "train_FD001.txt").unlink()
    with pytest.raises(data.ChecksumError, match="mismatch: RUL_FD001.txt") as err:
        data.verify(raw_dir, manifest)
    assert "missing: train_FD001.txt" in str(err.value)


def test_data_checksum_changes_when_data_changes(raw_dir: Path) -> None:
    before = data.data_checksum("FD001", raw_dir)
    assert before == data.data_checksum("FD001", raw_dir)
    (raw_dir / "RUL_FD001.txt").write_text("13\n30\n")
    assert data.data_checksum("FD001", raw_dir) != before


def test_committed_manifest_covers_all_subsets() -> None:
    names = set(data.read_manifest(config.CHECKSUM_FILE))
    for subset in config.SUBSETS:
        assert {f"train_{subset}.txt", f"test_{subset}.txt", f"RUL_{subset}.txt"} <= names


# --- extraction and download --------------------------------------------------------


def test_extract_members_unpacks_nested_zip_and_ignores_paths(tmp_path: Path) -> None:
    payload = nested_zip({"../../evil/RUL_FD001.txt": b"1\n", "other.pdf": b"x"})
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        written = data.extract_members(archive, {"RUL_FD001.txt"}, tmp_path)
    assert written == ["RUL_FD001.txt"]
    assert (tmp_path / "RUL_FD001.txt").read_bytes() == b"1\n"
    assert not (tmp_path / "other.pdf").exists()


def test_download_skips_network_when_data_is_valid(
    raw_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = make_manifest(raw_dir, ["RUL_FD001.txt"])

    def no_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("network should not be used")

    monkeypatch.setattr(data.urllib.request, "urlopen", no_network)
    data.download(raw_dir, manifest_path=manifest)


def test_download_rejects_zip_with_wrong_checksum(
    tmp_path: Path, raw_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = make_manifest(raw_dir, ["RUL_FD001.txt"])
    payload = nested_zip({"RUL_FD001.txt": b"12 \n30 \n"})
    monkeypatch.setattr(data.urllib.request, "urlopen", lambda *a, **k: FakeResponse(payload))
    with pytest.raises(data.ChecksumError, match="pinned"):
        data.download(tmp_path / "fresh", expected_sha256="0" * 64, manifest_path=manifest)


def test_download_extracts_and_verifies(
    tmp_path: Path, raw_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = make_manifest(raw_dir, ["RUL_FD001.txt"])
    payload = nested_zip({"RUL_FD001.txt": (raw_dir / "RUL_FD001.txt").read_bytes()})
    monkeypatch.setattr(data.urllib.request, "urlopen", lambda *a, **k: FakeResponse(payload))
    target = tmp_path / "fresh"
    data.download(target, expected_sha256=data.sha256_bytes(payload), manifest_path=manifest)
    assert data.load_test_rul("FD001", target).to_dict() == {1: 12, 2: 30}
