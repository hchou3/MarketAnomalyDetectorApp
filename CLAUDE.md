# Market Anomaly Detector — Claude Instructions

Detects market-wide stress regimes from macro instruments, explains why, and shows
instrument behavior during past anomaly episodes.

**Read before starting any task:**
1. `docs/architecture-plan.md` — full product scope, architecture, phase order, working rules
2. The current phase plan in `docs/phases/` (e.g., `phase0-plan.md`)

## Working rules

- Work on a feature branch per phase; open a PR. Never push directly to `main` unless asked.
- Don't commit generated files (`__pycache__`, artifacts, ONNX models, gRPC stubs, `.env`).
- Never commit API keys or credentials; read them from environment variables or Kubernetes Secrets.
- Write tests alongside code; every edge case in `docs/architecture-plan.md` section 5 gets a test.
- Report metrics exactly as measured, including when worse than before. Always show trivial baselines.
- If a task needs an open decision (section 9 of the architecture plan), ask before choosing.
- If a request conflicts with scope (section 1), point that out before building it.
- Keep `docs/architecture-plan.md` current: update section 3 after each phase and log decisions.

## Commands

```bash
# Install (editable, with dev tools)
pip install -e ".[dev]"

# Run tests
pytest

# Train model
mad-train

# Lint
ruff check .
ruff format --check .
```
