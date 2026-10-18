# Market Anomaly Detector

Detects market-wide stress regimes ("anomalies") from a set of macro instruments,
explains which features drove each detection, and shows how those instruments behaved
during past anomaly episodes. See [`docs/architecture-plan.md`](docs/architecture-plan.md)
for the product scope and the phased build plan (Phases 0–8).

> **Informational only.** This project produces historical regime scores and
> explanations. It does not provide investment advice, buy/sell/hold recommendations,
> or any suggestion that a result is safe to trade on.

## Current results

A walk-forward backtest over 2008 to April 2021 (14 yearly folds, 695 unseen weeks)
gives the ensemble a pooled F1 of 0.63 against 0.45 for the original Isolation Forest
baseline and 0.36 for always flagging an anomaly. It misses 2008 entirely and raises
22 false alarms in 2020. Details:
[`docs/reports/2026-10-10-walk-forward-backtest.md`](docs/reports/2026-10-10-walk-forward-backtest.md).

## Data

The model trains on weekly macro data (2000–2021) identified by Bloomberg codes.
The source and license of that CSV are unknown; read [`data/README.md`](data/README.md)
before redistributing it.

## Setup

Requires Python 3.12 or 3.13 and [`uv`](https://docs.astral.sh/uv/).

```bash
# Install the exact versions the published metrics were measured with
uv sync --frozen --extra dev

# Or with pip, using the same pins
pip install -r requirements-dev.txt
pip install -e . --no-deps
```

Results depend on library versions (calibration choice and F1 change across
scikit-learn releases), so use the locked versions when comparing metrics.

## Database setup

The data layer requires Postgres.  Start a local instance with Docker Compose, run
the Alembic migrations, and populate the instrument registry before running any
`mad-data` command.

```bash
docker compose up -d db
uv run alembic upgrade head      # create all tables (run once, or after a migration)
mad-data sync-registry           # load config/instruments.yaml → instruments + symbol_map
mad-data load-ews                # load data/raw/ews_weekly_2000_2021.csv → prices + labels
mad-data backfill                # fetch full history from Yahoo / FRED / Cboe
mad-data update                  # fetch incremental data (last date - 5 days, catches revisions)
mad-data status                  # last run summary, stale instruments, coverage, quality flags
mad-data report                  # write docs/reports/symbol-coverage.md and data/reports/symbol-coverage.csv
mad-data parity-stress           # verify DB stress series match data/raw/stress_extra.csv
```

`mad-data` commands exit with code 2 and a clear message if migrations have not been
run yet.  Set `MAD_DATABASE_URL` (default: `postgresql+psycopg://mad:mad@localhost:5432/mad`)
or pass `--database-url` to override.

`mad-data parity-stress` exits 1 if any non-revised series differs from the snapshot
beyond 1e-6 or has missing weeks; differences in revised series (NFCI, STLFSI) are
reported without failing.

For tests that require a live database:

```bash
docker compose up -d db
export MAD_TEST_DATABASE_URL=postgresql+psycopg://mad:mad@localhost:5432/mad_test
pytest -m db -v
```

## Usage

```bash
mad-train                      # single-split training; writes artifacts/pipeline.joblib + .meta.json
mad-train --calibration sigmoid
mad-backtest                   # walk-forward backtest; writes artifacts/backtest/

pytest                         # tests
ruff check . && ruff format --check .
```

`artifacts/` is git-ignored. Paths can be overridden with `MAD_DATA_PATH` and
`MAD_ARTIFACT_DIR`.

## Project layout

```
src/market_anomaly_detector/   library package (data, splits, pipeline, explain, train, backtest)
tests/                         pytest suite
data/raw/                      EWS weekly data and extra stress series
scripts/                       one-off data scripts (fetch_stress_data.py, retired in Phase 1)
docs/                          architecture plan, phase plans, reports, archived specs
notebooks/prototype.ipynb      archived exploratory notebook
```
