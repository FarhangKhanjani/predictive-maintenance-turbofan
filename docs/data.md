# Dataset: NASA C-MAPSS Turbofan Engine Degradation Simulation

## Source and terms

- **Official page:** NASA Prognostics Center of Excellence (PCoE) Data Set Repository, entry 6, "Turbofan Engine Degradation Simulation":
  https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
- **Download (linked from the NASA page, hosted by the PHM Society on S3):**
  https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip
  (12,429,152 bytes, SHA-256 `c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2`, checked 2026-10-25)
- **Terms:** NASA states no formal license. The page says users employ the data at their own risk, NASA and the data donors accept no liability, and publications should acknowledge the repository and the donors. We do not redistribute the data: `uv run rul-data download` fetches it from the source.

## Versioning and reproducibility

The source zip is public, about 12 MB, and has not changed since 2022, so we use the lightest option that is still verifiable instead of DVC:

1. The downloaded zip must match a pinned SHA-256 (`src/rul/config.py`).
2. Every extracted file must match the committed manifest [`data/SHA256SUMS`](../data/SHA256SUMS) (standard `sha256sum` format, so `cd data/raw && sha256sum -c ../SHA256SUMS` also works on Linux).
3. `uv run rul-data verify` re-checks files on disk at any time.
4. Each training run logs a combined checksum of the subset it used (`rul.data.data_checksum`), so a model can be traced back to its exact input bytes.

DVC (or lakeFS, or a versioned bucket) becomes the right tool once data changes over time or is produced by the pipeline itself, for example new engine telemetry or derived feature tables.
- **Citation:** A. Saxena and K. Goebel (2008). "Turbofan Engine Degradation Simulation Data Set", NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.
- **Method paper:** A. Saxena, K. Goebel, D. Simon, N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", PHM08, Denver CO, 2008 (included in the archive).

## Structure

| Subset | Train engines | Test engines | Operating conditions | Fault modes |
|---|---|---|---|---|
| FD001 | 100 | 100 | 1 (sea level) | 1 (HPC degradation) |
| FD002 | 260 | 259 | 6 | 1 (HPC) |
| FD003 | 100 | 100 | 1 | 2 (HPC, fan) |
| FD004 | 248 | 249 | 6 | 2 (HPC, fan) |

Each subset has three whitespace-separated text files, 26 columns, no header:
`unit, cycle, op_setting_1..3, sensor_1..21`. (NASA's readme says "sensor 26", which is a typo: 26 columns, 21 sensors.)

- `train_FD00x.txt`: every engine runs until failure. RUL at a row = last cycle of that engine minus current cycle.
- `test_FD00x.txt`: each engine stops at an unknown point before failure.
- `RUL_FD00x.txt`: true RUL at the last test cycle, one line per test engine, in unit order.

Every engine starts with a different, unknown amount of initial wear and manufacturing variation, and all sensors include noise. The data is simulated, not from real flights.

## Project scope

FD001 is the main benchmark. FD004 is an optional extension (needs normalization per operating condition).
