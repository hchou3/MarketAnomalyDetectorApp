# Instruction: Close out Phase 1 before starting Phase 2

You are picking up the `market-anomaly-detector` project. Phase 1 (see `phase1-build-instructions.md`) is mostly built: the two-stage cascade, `score(df)`, the time-based split and the leakage test all work, and all 10 tests pass. A verification pass found **three gaps that must be closed before Phase 2**, plus one repo-hygiene fix. This document is the full scope of the work.

Do these as four separate commits on a new branch `phase1-fixes`, then open one PR against `main`.

---

## Fix 1 — Per-prediction SHAP values are attached to the wrong feature names (bug)

### What's wrong
In `market_anomaly_detector/explain.py`, `SHAPExplainer.top_features()` assumes `TreeExplainer.shap_values()` returns either a list `[class_0, class_1]` or a 2-D array. Current SHAP versions (0.45+) return a **3-D array shaped `(n_rows, n_features, n_classes)`** — for one row that's `(1, 9, 2)`.

The code falls into the `else` branch and calls `.flatten()`, producing 18 values interleaved as `[f0_c0, f0_c1, f1_c0, f1_c1, ...]`. It then `zip`s those with the 9 feature names, so every feature after the first gets another feature's value or the wrong class's value. Symptom: the top features come in exact ± pairs, e.g. `XAU BGNL_cross: -0.1123` and `XAU BGNL: +0.1123`, when the real class-1 value of -0.1123 belongs to `VIX`.

The existing test (`test_score_output_contract`) only checks the dict's shape, so it passes despite this.

### What to do
1. In `top_features()`, normalise the SHAP output to a 1-D vector of **class-1 (anomaly) contributions, one per feature**, handling all three shapes:
   - `list` → take element `[1]`
   - 3-D `ndarray` `(n_rows, n_features, n_classes)` → take `[..., 1]`
   - 2-D `ndarray` `(n_rows, n_features)` → use as-is
   Then take row 0 for the single-row case.
2. Add a hard check: if the resulting vector's length ≠ `len(self.feature_names)`, raise `ValueError` with both lengths in the message. A silent mismatch is how this bug survived; it must be impossible to silently zip mismatched lengths again (use `zip(..., strict=True)` as well).
3. Leave the rest of the behaviour unchanged: still explain the Random Forest base learner, still rank by absolute impact, still return `[{"feature": str, "impact": float}, ...]`.

### Tests to add (`tests/test_explain.py`)
- **Correct labelling:** for a few rows of the val split, call `pipeline.explainer.explainer.shap_values(X_aug)` directly, extract the class-1 column yourself, and assert that every `{"feature", "impact"}` pair returned by `top_features()` matches that direct computation for that feature name (`pytest.approx`, `abs=1e-6`).
- **Unique names:** feature names in `top_features(X_row, n=len(AUG_FEATURE_COLS))` are all distinct and equal to the set `AUG_FEATURE_COLS`.
- **Length guard:** constructing a mismatch (e.g. an explainer with a truncated `feature_names` list) raises `ValueError`.
- **Shape handling:** a unit test that feeds the normalisation helper a list, a 2-D array and a 3-D array of known values and checks the output. Extract the normalisation into a small private function (e.g. `_class1_values(shap_vals)`) so this can be tested without SHAP.

---

## Fix 2 — Calibration method was never compared (missing requirement)

### What's wrong
The Phase 1 instructions say: *"Wrap it in `CalibratedClassifierCV` (sigmoid or isotonic — try both on the validation split, keep whichever calibrates better)."* `pipeline.py` hard-codes `method='sigmoid'`; isotonic was never tried.

A manual check during verification gave:

| Method | Val Brier (in-sample) | Test Brier | Test F1 |
|---|---|---|---|
| sigmoid | 0.105 | 0.148 | 0.836 |
| isotonic | 0.082 | 0.236 | 0.779 |

The val Brier above is **in-sample**: each calibrator was fit on the validation set and scored on the same validation set. Isotonic is much more flexible, so it wins in-sample and then loses badly on test. That's overfitting on a split with only 16 positives out of 104 rows. Choosing by in-sample val Brier would pick the wrong method. **Choosing by test results would be leakage.** Neither is acceptable.

### What to do
1. Add a `calibration` parameter to `AnomalyPipeline.__init__` with values `"auto"` (default), `"sigmoid"` or `"isotonic"`.
2. In `fit()`, when `calibration == "auto"`, select the method using **out-of-fold predictions on the validation split only**:
   - Use `StratifiedKFold(n_splits=5, shuffle=False)` over `X_v_aug, y_val`.
   - For each method, for each fold: fit `CalibratedClassifierCV(FrozenEstimator(self._stacking), method=m)` on the training folds of val and predict the held-out fold.
   - Compute the out-of-fold Brier score (primary) and log loss (secondary) for each method.
   - Pick the method with the lower OOF Brier score. Break ties in favour of `sigmoid` (fewer parameters).
3. Refit the chosen method on the **full** validation split as `self.calibrated`, exactly as now.
4. Store the decision on the fitted pipeline:
   - `self.calibration_method_` (the chosen method)
   - `self.calibration_report_`, a plain dict like `{"sigmoid": {"oof_brier": ..., "oof_log_loss": ...}, "isotonic": {...}}`
   When a method is forced, still compute and store the report so it is always available.
5. **Never** touch the test split inside `fit()`.

### Changes to `train.py`
- Add a `--calibration {auto,sigmoid,isotonic}` CLI flag (default `auto`) and pass it through.
- After fitting, print the calibration report and the chosen method.
- For information only, also print test Brier for **both** methods. Label this block clearly as `(reporting only — not used for selection)`. This keeps the comparison visible without letting it drive the choice.

### Tests to add (`tests/test_pipeline.py`)
- `calibration_method_` is in `{"sigmoid", "isotonic"}` after `fit()` with `"auto"`.
- `calibration_report_` has both methods with finite `oof_brier` values.
- Forcing `calibration="isotonic"` results in `calibration_method_ == "isotonic"`.
- An invalid value (e.g. `"platt"`) raises `ValueError` in `__init__`.

Keep the test fixture fast. The existing module-scoped fixture is fine; reuse it rather than fitting extra pipelines where possible.

---

## Fix 3 — The saved artifact is not reproducible (missing requirement for Phase 2)

### What's wrong
The repo has no `requirements.txt`. The committed `artifacts/pipeline.joblib` was built with scikit-learn **1.6.1** and an older SHAP whose pickled explainer imports `numba`. In a fresh Python 3.13 environment with current packages, `AnomalyPipeline.load("artifacts/pipeline.joblib")` fails with `ModuleNotFoundError: No module named 'numba'`, plus `InconsistentVersionWarning`s for every estimator. Phase 2's Docker image will hit exactly this.

### What to do
1. Create `requirements.txt` at the repo root with exact pins. This set is known to pass all tests and train successfully on Python 3.13:
   ```
   numpy==2.5.3
   scipy==1.18.1
   pandas==3.0.5
   scikit-learn==1.9.1
   shap==0.53.0
   joblib==1.6.0
   ```
   Put `pytest==9.1.1` in a separate `requirements-dev.txt` that starts with `-r requirements.txt`.
   Note that `sklearn.frozen.FrozenEstimator` requires scikit-learn ≥ 1.6, so never downgrade below that.
2. Verify in a **clean** virtual environment: `python -m venv .venv`, install `requirements-dev.txt`, then run the full test suite. Don't rely on whatever is already installed globally. If any pin fails to install, adjust it to the nearest working version and record why in the PR description.
3. Retrain the artifact in that clean environment after Fixes 1 and 2 are in:
   `python -m market_anomaly_detector.train`
   and commit the new `artifacts/pipeline.joblib`.
4. Write a sidecar file `artifacts/pipeline.meta.json` from `train.py` alongside the artifact, containing:
   - the versions of python, numpy, pandas, scikit-learn, shap and joblib at train time
   - the train/val/test date ranges
   - the chosen calibration method and calibration report
   - the val and test F1/precision/recall for the ensemble, and the baseline test F1
   - a UTC timestamp
5. In `AnomalyPipeline.load()`, if the sidecar exists, compare its scikit-learn and shap versions against the running versions and emit a `warnings.warn(...)` naming both versions when they differ. Do not raise; just make the mismatch loud and explicit.
6. Add a test that saves a freshly fitted pipeline to a temp dir, reloads it with `AnomalyPipeline.load()`, and checks that `score()` on the same input gives an identical result before and after the round-trip.

---

## Fix 4 — Stop committing bytecode caches (hygiene)

`__pycache__/` folders are committed, so every test run leaves the working tree dirty.

1. Add the following to `.gitignore`, keeping the existing `Claude Instructions` line:
   ```
   __pycache__/
   *.pyc
   .pytest_cache/
   .venv/
   ```
2. Remove the cached files from the index without deleting anything else:
   `git rm -r --cached market_anomaly_detector/__pycache__ market_anomaly_detector/tests/__pycache__`
3. Confirm `git status` is clean after running the test suite.

Do this as the **first** commit so the later commits don't pick up `.pyc` noise.

---

## Explicitly out of scope

Do not change any of the following in this PR. They are either correct as-is or belong to Phase 2+:

- The `score(df) -> dict` signature or return shape. Phase 2 calls it directly.
- The feature set (`FEATURE_COLS`), feature engineering, or the 9y/2y/1y time-based split.
- Model hyperparameters or ensemble structure.
- The SHAP target (it explains the RF base learner; leave that decision alone).
- `api/api.py` (Flask stub), FastAPI, Redis, Docker, batch scoring. All Phase 2.

If you notice something else wrong outside this scope, mention it in the PR description rather than fixing it.

---

## Acceptance criteria

The PR is done when all of these hold:

- [ ] Commit order: (1) gitignore/cache removal, (2) SHAP fix + tests, (3) calibration selection + tests, (4) requirements, meta sidecar, version warning, round-trip test and retrained artifact.
- [ ] Full test suite passes in a clean venv built only from `requirements-dev.txt`.
- [ ] `git status` is clean after running the tests.
- [ ] `AnomalyPipeline.load("artifacts/pipeline.joblib")` works in that clean venv with no `InconsistentVersionWarning` and no `numba` import.
- [ ] `score()` top features no longer show ± mirrored pairs, and each impact matches a direct SHAP computation for that feature.
- [ ] `artifacts/pipeline.meta.json` exists and contains everything listed in Fix 3.
- [ ] The PR description includes the **actual measured** output of `train.py`: baseline test F1, ensemble val and test F1/precision/recall, the calibration report and the method chosen. Report the numbers you measure. Do not copy the numbers from this document, and do not round them up. If the ensemble no longer beats the baseline after these changes, say so plainly.
