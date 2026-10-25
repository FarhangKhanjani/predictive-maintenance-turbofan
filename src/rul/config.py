"""Central configuration: paths, dataset source, and column layout.

Values that differ between machines (for example the data directory inside a
Docker container) can be overridden with environment variables, see .env.example.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("RUL_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
CHECKSUM_FILE = DATA_DIR / "SHA256SUMS"

# NASA PCoE Data Set Repository, entry 6. The NASA page links to this S3 file.
DATASET_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)
DATASET_ZIP_SHA256 = "c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2"

SUBSETS = ("FD001", "FD002", "FD003", "FD004")
SPLITS = ("train", "test")

N_SETTINGS = 3
N_SENSORS = 21
SETTING_COLUMNS = [f"op{i}" for i in range(1, N_SETTINGS + 1)]
SENSOR_COLUMNS = [f"s{i}" for i in range(1, N_SENSORS + 1)]
ID_COLUMNS = ["unit", "cycle"]
COLUMNS = ID_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS
