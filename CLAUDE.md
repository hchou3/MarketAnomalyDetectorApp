# Market Anomaly Detector — Claude Instructions

Detects market-wide stress regimes from macro instruments, explains why, and shows
instrument behavior during past anomaly episodes.

**Read before starting any task:**
1. `docs/architecture-plan.md`: scope, architecture, phase order, current state, decisions
2. The current phase plan in `docs/phases/` (`phase1-plan.md` is next)
3. `docs/reports/` for the latest measured results

## Working rules

- Before a phase, write its plan from an audit of the code; get the user's go-ahead first.
- Audit every branch, not just `main` (`git branch -a`, `git log main..origin/<branch>`).
- One feature branch per phase or step; open a PR. Never push to `main` unless asked.
- Running code must leave `git status` clean; fix ignore rules instead of committing outputs.
- Don't commit generated files (`__pycache__`, `artifacts/`, ONNX models, gRPC stubs, `.env`).
- Never commit API keys or credentials; read them from environment variables or Secrets.
- Write tests alongside code; every edge case in plan section 5 gets a test.
- Report metrics exactly as measured, with trivial baselines, using the locked versions.
- If a task needs an open decision (plan section 9), ask before choosing.
- If a request conflicts with scope (plan section 1), say so before building it.
- Keep `docs/architecture-plan.md` current: section 3 after each phase, decisions in section 8.
- This workspace can't reach Yahoo, FRED, or Cboe; build against recorded fixtures and
  ask the user to run live fetches.

## Commands

```bash
uv sync --frozen --extra dev   # install locked versions
pytest
mad-train
mad-backtest
ruff check . && ruff format --check .
```

## Commit rules

- Do not add Co-Authored-By or other attribution lines to commits.
