# Market Anomaly Detector

Detects market-wide stress regimes ("anomalies") from a set of macro instruments,
explains which features drove the detection, and shows how those instruments behaved
during past anomaly episodes. See [`docs/architecture-plan.md`](docs/architecture-plan.md)
for the full product scope and six-phase build plan.

> **Informational only.** This project produces historical regime scores and
> explanations. It does not provide investment advice, buy/sell/hold recommendations,
> or any suggestion that a result is safe to trade on.

## Data

The model trains on weekly macro data (2000–2021) including gold (XAU), VIX, the
US Dollar Index (DXY), Italian 2-year yields (GTITL2YR), and EONIA. The source and
license of the CSV are currently unknown — see [`data/README.md`](data/README.md)
before redistributing.

## Setup

Requires Python ≥ 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
# Install all dependencies from the lock file
uv sync --frozen --extra dev

# Or with plain pip (installs without lock-file pinning)
pip install -e ".[dev]"
```

## Usage

```bash
# Train the model (saves artifact to artifacts/pipeline.joblib)
mad-train

# Override data or artifact path
mad-train --data path/to/data.csv --out path/to/artifact.joblib

# Run tests
pytest

# Lint
ruff check .
ruff format --check .
```

## Project layout

```
src/market_anomaly_detector/   # library package
tests/                         # pytest test suite
data/raw/                      # training data
docs/                          # architecture plan and phase plans
notebooks/prototype.ipynb      # archived exploratory notebook
```

## Links

- Architecture and phase plan: [`docs/architecture-plan.md`](docs/architecture-plan.md)
- Phase 0 (this branch): [`docs/phases/phase0-plan.md`](docs/phases/phase0-plan.md)
