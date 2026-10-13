# Phase 0 Plan — Repo Hygiene

**Parent plan:** `architecture-plan.md`, section 6, Phase 0.
**Written:** 2026-10-13, against `main` at `d2fab72`.
**Branch:** `phase-0/repo-hygiene` → one PR into `main`.

Phase 0 changes structure and tooling only. Model behavior, features, and metrics
must be identical before and after; anything that changes results belongs to Phase 2.

---

## 1. Audit of the current codebase

Checked on 2026-10-13 by cloning `main`, running the tests, and running `train.py`.

| # | Finding | Evidence | Fix in this phase |
|---|---|---|---|
| A1 | Bytecode is committed | 9 `__pycache__/*.pyc` files tracked; running any code makes git show changes | Untrack, ignore |
| A2 | Model artifact is committed | `artifacts/pipeline.joblib`, 12.9 MB, tracked | Untrack, ignore |
| A3 | `.gitignore` has one entry | Only `Claude Instructions` | Replace with a full Python/ML ignore file |
| A4 | Instructions folder is ignored but partly tracked | The 3 original files were committed before the ignore rule; new files (`architecture-plan.md`, this plan) are ignored, so a fresh clone gets outdated instructions | Move docs to `docs/`, track them, add `CLAUDE.md` |
| A5 | No dependency manifest | No `requirements.txt` or `pyproject.toml`; code needs scikit-learn ≥ 1.6 for `sklearn.frozen.FrozenEstimator` | Add `pyproject.toml` + lock file |
| A6 | Tests rely on `sys.path.insert` hacks | Both test files prepend the repo root to `sys.path` | Install the package in editable mode; remove the hacks |
| A7 | Tests live inside the package | `market_anomaly_detector/tests/` ships with the library | Move to top-level `tests/` |
| A8 | Dead Flask stub | `api/api.py` is Flask; `upload_csv(request)` would fail at runtime; Phase 4 replaces it with FastAPI | Delete `api/` |
| A9 | Broken prompt template | `prompts/correlations_analysis.j2` uses an f-string placeholder inside Jinja and asks the LLM to rank tickers; out of scope | Delete `prompts/` |
| A10 | Data file name has spaces and a Google Sheets export suffix | `data/FinancialMarketData.xlsx - EWS.csv` | Rename to `data/raw/ews_weekly_2000_2021.csv`; record origin in `data/README.md` |
| A11 | Prototype notebook in the instructions folder | `Market_Anomaly_Detector (3).ipynb`, 1.5 MB with outputs; no secrets found (keys are read via `google.colab.userdata`) | Move to `notebooks/prototype.ipynb`, outputs stripped |
| A12 | No README, no CI, no linter | — | Add README, GitHub Actions (lint + tests), ruff |
| A13 | Hard-coded paths relative to the package | `train.py` and tests build `data/...` paths from `__file__` | Central `paths.py`; overridable by env var |
| A14 | `print`-based logging in library code | `pipeline.py` `save()`/`evaluate()`, `train.py` | Switch library code to `logging`; CLI keeps printing via the logger |

Baseline to preserve (record in the PR): `pytest` → 10 passed;
`python -m market_anomaly_detector.train` → test F1 0.8358, baseline F1 0.7778.
Measured with Python 3.13, pandas 3.0.5, scikit-learn 1.9.1.

Not fixed in Phase 0 (by design): raw-level features, the 12-year split, SHAP on the
RF only, the weak `f1 > 0.3` test. All are Phase 2.

---

## 2. Target layout

```
.
├── CLAUDE.md                     # short pointer: read docs/architecture-plan.md first
├── README.md
├── pyproject.toml
├── uv.lock                       # or requirements-lock.txt (see decision D3)
├── .gitignore
├── .github/workflows/ci.yml
├── docs/
│   ├── architecture-plan.md      # moved from Claude Instructions/
│   ├── phases/phase0-plan.md     # this file
│   └── archive/
│       ├── market-anomaly-detector-project.md
│       └── phase1-build-instructions.md
├── data/
│   ├── README.md                 # origin, license status, columns, label notes
│   └── raw/ews_weekly_2000_2021.csv
├── notebooks/
│   └── prototype.ipynb           # outputs stripped
├── src/market_anomaly_detector/
│   ├── __init__.py
│   ├── paths.py                  # new: repo-relative defaults + env overrides
│   ├── data.py
│   ├── splits.py
│   ├── pipeline.py
│   ├── explain.py
│   └── train.py
└── tests/
    ├── conftest.py               # shared fixtures (data path, featured df, trained pipeline)
    ├── test_pipeline.py
    └── test_splits.py
```

`src/` layout means tests import the installed package, so a missing dependency or
packaging mistake fails in CI instead of passing by accident.

---

## 3. Tasks (one commit each, in order)

1. **Branch and baseline.** Create `phase-0/repo-hygiene`. Run tests and `train.py`;
   paste the numbers into the PR description.
2. **`.gitignore` + untrack generated files.** Full ignore list: `__pycache__/`,
   `*.py[cod]`, `.pytest_cache/`, `.ruff_cache/`, `.venv/`, `venv/`, `*.egg-info/`,
   `build/`, `dist/`, `.env`, `.env.*`, `artifacts/`, `*.joblib`, `*.onnx`,
   `model_repository/`, `.ipynb_checkpoints/`, `.DS_Store`, `.idea/`, `.vscode/`,
   generated gRPC stubs (`*_pb2.py`, `*_pb2_grpc.py`). Then
   `git rm -r --cached` the `__pycache__` folders and `artifacts/pipeline.joblib`.
   Remove the `Claude Instructions` ignore line.
3. **Docs move.** `git mv` the instruction docs into `docs/` per section 2; add
   `docs/architecture-plan.md` and `docs/phases/phase0-plan.md` (currently untracked);
   update path references inside them. Add `CLAUDE.md` (under 40 lines): project
   one-liner, "read `docs/architecture-plan.md` and the current phase plan before
   working," the working rules from section 7 of the plan, and the test/lint commands.
4. **Delete dead code.** Remove `api/` and `prompts/`.
5. **Notebook.** `git mv` to `notebooks/prototype.ipynb`; strip outputs
   (`nbstripout` or `jupyter nbconvert --clear-output`). Add a header cell saying it's
   the archived prototype and not used by the app.
6. **Data.** `git mv` the CSV to `data/raw/ews_weekly_2000_2021.csv`. Write
   `data/README.md`: column list (Bloomberg codes), weekly Tuesday dates, 2000-01-11 to
   2021-04-20, 1,111 rows, `Y` = anomaly label with unknown labeling rule, origin and
   license unknown (open decision 2 in the plan).
7. **Packaging.** Move the package to `src/`. Add `pyproject.toml`:
   - `requires-python = ">=3.12"`
   - runtime: `numpy`, `pandas`, `scikit-learn>=1.6`, `shap`, `joblib`
     (lower bounds only; exact versions live in the lock file)
   - dev group: `pytest`, `ruff`, `nbstripout`
   - console script: `mad-train = market_anomaly_detector.train:main`
   - ruff config (line length 100, import sorting on)
   - pytest config (`testpaths = ["tests"]`)
   Generate the lock file.
8. **Paths and logging.** Add `paths.py` with `REPO_ROOT`, `DATA_PATH`
   (default `data/raw/ews_weekly_2000_2021.csv`, override `MAD_DATA_PATH`),
   `ARTIFACT_DIR` (default `artifacts/`, override `MAD_ARTIFACT_DIR`). Use it in
   `train.py` and tests. Replace `print` in library code with `logging`.
9. **Tests.** Move to top-level `tests/`; delete the `sys.path` hacks; move the
   shared data/pipeline fixtures to `conftest.py` with `scope="session"` so the
   pipeline trains once per run. No assertion changes.
10. **Lint.** Run `ruff check --fix` and `ruff format`. Formatting-only changes;
    confirm tests and training metrics are unchanged.
11. **CI.** `.github/workflows/ci.yml` on push and PR: set up Python 3.12 and 3.13,
    install from the lock file, `ruff check`, `ruff format --check`, `pytest`.
12. **README.** What the project is (one paragraph, matching plan section 1), setup,
    `pytest`, `mad-train`, link to `docs/architecture-plan.md`, data caveats, and the
    "informational, not investment advice" line.
13. **Update the plan.** In `docs/architecture-plan.md`: update section 3 (current
    state), add a decision-log entry for the new layout, update file paths that changed.

---

## 4. Acceptance checks

Run on a fresh clone of the PR branch:

- [ ] `git ls-files | grep -E '__pycache__|\.pyc$|\.joblib$'` returns nothing
- [ ] Install from the lock file into a clean venv succeeds
- [ ] `pytest` → 10 passed, no `sys.path` manipulation anywhere (`grep -r "sys.path" src tests`)
- [ ] `mad-train` reproduces test F1 0.8358 and baseline F1 0.7778
- [ ] Running tests or training leaves `git status` clean
- [ ] `ruff check` and `ruff format --check` pass
- [ ] CI is green on both Python versions
- [ ] `CLAUDE.md` and `docs/architecture-plan.md` are tracked and present on a fresh clone
- [ ] Notebook has no outputs (`grep -c '"outputs": \[\]'` equals the code-cell count)

---

## 5. Decisions (defaults Claude will use unless the user says otherwise)

| # | Decision | Default |
|---|---|---|
| D1 | Track the instruction docs in git? The repo is public, so they become public. | Yes, under `docs/` |
| D2 | Keep the prototype notebook in the repo? | Yes, in `notebooks/`, outputs stripped |
| D3 | Lock tool | `uv` (`uv.lock`); fall back to `pip-tools` if the user prefers |
| D4 | Rewrite git history to purge the 12.9 MB `pipeline.joblib`? | No. Untrack only. Only two commits exist, so a rewrite is cheap if the user wants it, but it changes commit hashes. |
| D5 | Keep the CSV in the repo while its license is unknown? | Yes for now, flagged in `data/README.md`; revisit with open decision 2 |

---

## 6. Risks

- **`src/` move breaks imports** in anything outside the repo that imports
  `market_anomaly_detector` by path. Nothing does today.
- **Old artifact incompatibility:** the untracked `pipeline.joblib` was pickled with a
  specific scikit-learn version. After Phase 0, regenerate it with `mad-train` instead
  of loading the old one.
- **Lock file vs. local Python:** the existing bytecode shows Python 3.13 locally; CI
  tests both 3.12 and 3.13 so the lock file must resolve for both.
