# Phase 0 Plan — Repo Hygiene

**Parent plan:** `docs/architecture-plan.md`, section 6, Phase 0.
**Written:** 2026-10-13. **Revised:** 2026-10-13 after finding unmerged work on
`phase1-fixes` and `walk-forward-backtest` (the first version audited `main` only).
**Status:** complete on `phase-0/repo-hygiene-v2`, pending PR review (section 7).

Phase 0 changes structure and tooling only. Model behavior, features, and metrics
must be identical before and after; anything that changes results belongs to Phase 2.

---

## 0. Merge existing work first

Two branches held work from 2026-10-09/10 that wasn't on `main`:

| Branch | Commits | Contents |
|---|---|---|
| `phase1-fixes` | 9 | Untrack bytecode (Fix 4); SHAP class-1 fix + explain tests (Fix 1); OOF calibration selection + `--calibration` flag (Fix 2); pinned `requirements*.txt`, `.meta.json` sidecar, version warning on load (Fix 3); `fetch_stress_data.py` + `stress_extra.csv` (Cboe/FRED stress series) |
| `walk-forward-backtest` | 5 (4 shared) | `backtest.py`: expanding-window backtest 2008–2021 with pooled metrics, trivial baselines, PR/ROC AUC, year-level bootstrap; tests; results in `artifacts/backtest/` |

"phase1" in the branch name refers to Phase 1 of the *old* spec (the ensemble), not
Phase 1 of this plan. Both were merged into `main` on 2026-10-13.

---

## 1. Audit (state after step 0)

| # | Finding | Fix in this phase |
|---|---|---|
| A1 | ~~Bytecode committed~~ (fixed on `phase1-fixes`) | Verify only |
| A2 | `artifacts/` committed: `pipeline.joblib` (12.9 MB), `pipeline.meta.json`, `backtest/{summary.json, per_year.csv, predictions.csv}` | Untrack and ignore; keep findings as a report (A15) |
| A3 | Minimal `.gitignore` | Full Python/ML ignore file |
| A4 | Instructions folder ignored but partly tracked; new plans missing from clones | Docs in tracked `docs/`; `CLAUDE.md` |
| A5 | Pinned `requirements*.txt` but no packaging | `pyproject.toml` + `uv.lock` at the same versions |
| A6 | Tests use `sys.path.insert` | Editable install; remove |
| A7 | Tests inside the package | Top-level `tests/` |
| A8 | Dead Flask stub `api/api.py` | Delete |
| A9 | Broken `prompts/correlations_analysis.j2` | Delete |
| A10 | Data file names with spaces; CSVs at `data/` root | `data/raw/ews_weekly_2000_2021.csv`, `data/raw/stress_extra.csv`, `data/README.md` |
| A11 | 1.5 MB notebook with outputs in the instructions folder | `notebooks/prototype.ipynb`, outputs stripped |
| A12 | `fetch_stress_data.py` at repo root | `scripts/` |
| A13 | No README, CI, or linter | README, GitHub Actions, ruff |
| A14 | Paths built from `__file__` in several modules | `paths.py` with env overrides |
| A15 | Backtest results only as raw JSON/CSV | `docs/reports/2026-10-10-walk-forward-backtest.md` |
| A16 | `print` in library code | `logging` |

**Baseline rule:** measure on merged `main` with the pinned versions before any change;
the same numbers must reproduce at the end. Results differ across library versions.

---

## 2. Target layout

```
.
├── CLAUDE.md, README.md, pyproject.toml, uv.lock, requirements*.txt, .gitignore
├── .github/workflows/ci.yml
├── docs/
│   ├── architecture-plan.md
│   ├── phases/phase0-plan.md, phase1-plan.md
│   ├── reports/2026-10-10-walk-forward-backtest.md
│   └── archive/  (original spec, old phase-1 instructions, phase1-fixes instructions)
├── data/README.md, data/raw/ews_weekly_2000_2021.csv, data/raw/stress_extra.csv
├── notebooks/prototype.ipynb
├── scripts/fetch_stress_data.py
├── src/market_anomaly_detector/  __init__, paths, data, splits, pipeline, explain, train, backtest
└── tests/  conftest, test_pipeline, test_splits, test_explain, test_backtest
```

---

## 3. Tasks

1. Baseline on merged `main`.
2. `.gitignore`; untrack `artifacts/`.
3. Backtest report before the raw files leave git.
4. Docs to `docs/`; `CLAUDE.md`.
5. Delete `api/`, `prompts/`.
6. Notebook to `notebooks/`, outputs stripped.
7. Data to `data/raw/`; script to `scripts/`; `data/README.md`.
8. `src/` layout; `pyproject.toml`; `uv.lock` at the pinned versions; `mad-train`, `mad-backtest`.
9. `paths.py`; logging.
10. Tests to `tests/`; no `sys.path`; session fixtures in `conftest.py`.
11. ruff check and format; re-run baseline.
12. CI on 3.12 and 3.13 from the lock file.
13. README.
14. Update the architecture plan.

---

## 4. Acceptance checks

- [x] No `__pycache__`, `.pyc`, `.joblib`, or `artifacts/` files tracked
- [x] Clean install from the lock file (`uv sync --frozen --extra dev`) on 3.12 and 3.13
- [x] `pytest`: 32 passed, same as baseline; no `sys.path` manipulation in `src` or `tests`
- [x] `mad-train` and `mad-backtest` reproduce the baseline exactly
- [x] `git status` clean after tests, training, and backtest
- [x] `ruff check .` and `ruff format --check .` pass
- [ ] CI green on 3.12 and 3.13 (runs when the PR is opened)
- [x] `CLAUDE.md`, `docs/architecture-plan.md`, and phase plans tracked
- [x] Notebook has no outputs

---

## 5. Decisions

| # | Decision | Outcome |
|---|---|---|
| D0 | Land the two branches first | Merged into `main` by the user |
| D1 | Track docs in git (public repo) | Yes, under `docs/` |
| D2 | Keep the notebook | Yes, outputs stripped |
| D3 | Lock tool | `uv`; `requirements*.txt` kept for pip users |
| D4 | Rewrite history for `pipeline.joblib` | No; untracked only |
| D5 | Keep CSVs while license unknown | Yes, flagged in `data/README.md` |

---

## 6. Risks

- **Version-sensitive baselines.** Always measure with the locked versions.
- **Old artifacts** were pickled under specific versions; regenerate with `mad-train`.

---

## 7. Outcome (2026-10-13)

**Status: complete on `phase-0/repo-hygiene-v2`, pending PR review.**

How it went: the first Phase 0 branch (`phase-0/repo-hygiene`) was built from the old
`main`, before step 0, so it lacked the `phase1-fixes` and `walk-forward-backtest`
work. The user then reverted a stray build-output commit on `main` (`577a35c`),
merged both branches into `main`, and `phase-0/repo-hygiene-v2` re-applied the
Phase 0 branch on top with conflicts resolved. `phase-0/repo-hygiene` is superseded.

Baseline (merged `main`, pinned versions) vs. result (v2 branch):

| Check | Baseline | Result |
|---|---|---|
| Tests | 32 passed | 32 passed (Python 3.12 and 3.13, from `uv.lock`) |
| `train` calibration choice | isotonic | isotonic |
| `train` test F1 (baseline → ensemble) | 0.7778 → 0.7838 | identical (`.meta.json` metrics and calibration report equal) |
| Backtest pooled F1 (ensemble / baseline / always-flag) | 0.632 / 0.453 / 0.361 | identical; `predictions.csv` byte-identical |
| `ruff check .`, `ruff format --check .` | — | pass |

Conflict resolutions worth knowing:
- `pipeline.py`: kept OOF calibration selection from `phase1-fixes` with Phase 0's
  logging and formatting.
- `train.py`: Phase 0's `src/` version plus the `--calibration` flag, test-set
  calibration report, and `.meta.json` sidecar from `phase1-fixes`, all via `logging`.
- `tests/test_pipeline.py` auto-merged without conflict but was broken: new tests from
  `phase1-fixes` used imports Phase 0 had removed. Caught by ruff; fixed.

Deviations from the task list:
- `uv.lock` on the first branch resolved to the latest releases (scikit-learn 1.9.1,
  pandas 3.0.6), not the pins. Fixed with `[tool.uv] constraint-dependencies` matching
  `requirements.txt`. The pinned `shap 0.50.0` can't resolve on Python 3.14, so
  `requires-python` is now `>=3.12,<3.14`. `requirements*.txt` kept for pip users.
- E501 (line length) is exempted for `backtest.py` (long report f-strings, split in
  Phase 2) and `scripts/fetch_stress_data.py` (retired in Phase 1).
- `print` remains in `backtest.py`'s CLI report output; library code paths log.
- `data` optional dependencies (`yfinance`, `certifi`) not added; Phase 1 adds them
  with the providers.

Acceptance checks: all pass except "CI green," which runs when the PR is opened.
