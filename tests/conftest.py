"""Shared fixtures: tiny synthetic C-MAPSS files so tests never need the real dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rul import config


def write_cmapss_file(path: Path, lifetimes: dict[int, int], seed: int = 0) -> None:
    """Write a whitespace-separated file in C-MAPSS layout (26 columns, no header)."""
    rng = np.random.default_rng(seed)
    rows = []
    for unit, life in lifetimes.items():
        for cycle in range(1, life + 1):
            values = rng.normal(size=config.N_SETTINGS + config.N_SENSORS).round(4)
            rows.append(" ".join([str(unit), str(cycle), *map(str, values)]))
    path.write_text("\n".join(rows) + "\n")


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """A raw data directory with a tiny FD001: 2 train engines, 2 test engines."""
    raw = tmp_path / "raw"
    raw.mkdir()
    write_cmapss_file(raw / "train_FD001.txt", {1: 5, 2: 7})
    write_cmapss_file(raw / "test_FD001.txt", {1: 3, 2: 4}, seed=1)
    (raw / "RUL_FD001.txt").write_text("12 \n30 \n")
    return raw
