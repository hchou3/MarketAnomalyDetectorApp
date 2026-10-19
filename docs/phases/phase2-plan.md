# Phase 2 Plan — Model Rebuild

**Parent plan:** `docs/architecture-plan.md`, section 6, Phase 2 (specs in 4.3, 2.3).
**Written:** 2026-10-18, against `main` at `bcdd26d` (Phase 1 Steps 1–7 merged).
**Branches:** one branch and PR per step (`phase-2/<step>`), merged in order.

## Goal

A market-level stress model that is trained and served on the **same data the app can
fetch**, built from point-in-time features read from the database, evaluated honestly
at the **event level** against strong simple baselines, calibrated, explainable on its
final output, and exportable to ONNX for Triton (Phase 5).

**Not in Phase 2:** the API, Redis, gRPC, Triton deployment, episode analytics for the
UI (Phase 3), live/intraday models (Phase 8).

---

## 1. Audit of the current model (main at `bcdd26d`)

| # | Finding | Evidence | Consequence |
|---|---|---|---|
| M1 | Features are raw **levels** plus era flags | `data.py`: 5 Bloomberg series, 5/20-week MA crosses, `USD_Over_100` | Learns "which decade", not stress. Consistent with the 2008 miss (0/27 weeks) |
| M2 | Features come from the CSV, not the database | `train.py`, `backtest.py` call `load_data(DATA_PATH)` | Phase 1's point-in-time data isn't used yet |
| M3 | Trains on Bloomberg series the app can't serve | `GTITL2YR`, `XAU BGNL` have no exact live source | Train/serve skew the moment live scoring starts |
| M4 | Only a "detect" label | `Y` at week t | No "predict" head; no lead-time measurement |
| M5 | Row-level metrics only | `backtest.py` pooled P/R/F1, AUCs, bootstrap | Can't answer "did we catch 2008, and how early?" |
| M6 | Weak baselines only | always-flag, 3-feature IsolationForest | No test against the obvious rule ("VIX is high") |
| M7 | Calibration picked by Brier on thin windows | 3 of 14 folds chose isotonic with 8–10 positives | Isotonic overfits small windows; log loss 0.65 vs 0.40 once |
| M8 | **Calibration can't be exported to ONNX** | Probe 2026-10-18: IsolationForest and StackingClassifier export with max abs diff < 5e-7; `CalibratedClassifierCV(FrozenEstimator(...))` fails with `KeyError: FrozenEstimator` (skl2onnx 1.20, scikit-learn 1.6.1) | Blocks Triton unless calibration is restructured |
| M9 | SHAP explains the RF base learner, not the output | `explain.py` | Explanations can contradict the reported probability |
| M10 | Decision threshold hard-coded at 0.5 | `pipeline.py` lines 175, 201 | Arbitrary operating point; not tied to an alert budget |
| M11 | Model selection and reporting use the same folds | One walk-forward run produces the headline | Any tuning on it makes the headline optimistic |

**Baseline to beat** (`docs/reports/2026-10-10-walk-forward-backtest.md`, reproduced on
main 2026-10-18): pooled F1 0.632 (ensemble), 0.453 (IsolationForest), 0.361 (always
flag); ROC AUC 0.88; 2008 missed entirely; 22 false-alarm weeks in 2020.

---

## 2. Prerequisites (must be done before Step 2.1)

1. **Phase 1 close-out.** Commit `docs/reports/symbol-coverage.md` and align the YAML
   fidelity labels with it (`mad-data report --check` passes). Run
   `mad-data parity-stress` on live data. Replace `docs/phases/phase1-plan.md` with
   the project copy.
2. **Decisions L1, S1, F1** in section 6 answered or defaults accepted.

---

## 3. Design

### 3.1 Training data = serving data
Features are built from the **live providers' history** in the database (Yahoo, FRED,
Cboe), never from `bloomberg_ews`. Bloomberg data is used only for the labels and as
the fidelity reference. The model then sees in training exactly what it sees in
production, and the Phase 1 coverage report tells us how well each live series
stands in for the Bloomberg one.

Consequence: v1 can only use instruments whose live history starts by about 2000
(the first training year). Later series are a v2 experiment (3.4).

### 3.2 Feature spec and builder
- `features/spec.py`: a list of `FeatureSpec(name, inputs, transform, window, label)`.
  One module, read by training, scoring, and (Phase 8) the stream job.
- `features/build.py`: `build(session, spec, weeks, as_of=None) -> DataFrame`.
  - **Weekly anchor `W-TUE`** (matches the labels), configurable.
  - **Point in time:** for week t, an input value is usable only if
    `known_at <= t`. Daily series use the last value on or before t; weekly and
    monthly series use the latest *published* value.
  - Every transform uses trailing windows only. No `fillna` across more than
    one period; missing stays missing.
- **Transforms** (by registry `units`):
  - prices and indices: log return over 1, 4, 12 weeks; drawdown from trailing 52-week
    high; realized volatility of daily log returns over trailing 20 trading days
  - percent (yields, rates): change in basis points over 1, 4, 12 weeks; spreads
    (10y−3m, 10y−2y) as levels in bp
  - all: z-score vs trailing 156-week (3-year) window, min 104 weeks of history
- **No raw levels, no era flags.** Spreads and z-scores are the only level-like inputs.
- **Readable labels** on every feature ("VIX, 4-week change") for explanations.

### 3.3 Feature set v1 (live history back to ~2000)
| Instrument | Source | Notes |
|---|---|---|
| `vix` | Cboe / Yahoo | from 1990 |
| `msci_usa` | Yahoo `^GSPC` | index proxy, from 1927 |
| `dollar_index` | Yahoo `DX-Y.NYB` | |
| `usd_jpy`, `gbp_usd` | Yahoo (primary) / FRED | |
| `gold` | Yahoo `GC=F` | futures; roll jumps |
| `wti_front` | Yahoo `CL=F` | negative price 2020-04: log returns undefined that week |
| `us_10y_yield`, `us_2y_yield`, `us_3m_yield` | FRED | spreads 10y−3m, 10y−2y |
| `skew` | Cboe | from 1990 |
| `nfci`, `stlfsi` | FRED | **revised series**; included only per decision F1 |

Final inclusion requires coverage-report fidelity `exact` or `proxy`. Italian yields
(`GTITL2YR`) are dropped: no daily live source (closes open decision 3).

### 3.4 Feature set v2 (experiment, not required)
Adds late-starting series (VIX3M from 2006, VVIX, OVX, GVZ, `HYG`, `LQD`, `EMB`) with
missing values before their start. Requires a model that handles NaN (e.g.
`HistGradientBoostingClassifier`; check ONNX export first). Kept only if it improves
the development-fold event metrics.

### 3.5 Labels and episodes
- `labels.py`: `detect` = label at week t; `predict` = any label in weeks t+1..t+h
  (h = 4, configurable). The last h weeks of a label set have no `predict` target and
  are dropped from `predict` training.
- `episodes.py`: contiguous runs of label 1, merging runs separated by ≤ 2 calm weeks
  (configurable). Shared with Phase 3. Each episode: start, end, peak week, length.
- Label set is a parameter (`ews_v1` today), so a future label set (decision L1)
  drops in without code changes.

### 3.6 Evaluation harness v2 (extends `backtest.py`)
- **Folds:** unchanged expanding-window yearly scheme, built from the database
  feature frame.
- **Development vs final split:** model and feature choices are made on test years
  **2008–2017** only. Years **2018–2021** are the **final confirmation set**, run
  once at the end (Step 2.5) and reported as-is.
- **Event-level metrics** (per episode, then aggregated):
  - detect: episode caught (any alert inside it), weeks from onset to first alert
  - predict: lead time = weeks before onset of the first alert in [onset − h, onset]
  - false-alarm episodes: contiguous alert runs not overlapping any episode ± 2 weeks,
    reported **per calm year**
- **Row-level and probability metrics** kept: P/R/F1, PR AUC, ROC AUC, Brier,
  log loss, reliability curve, expected calibration error.
- **Baselines**, all run through the same harness:
  always-flag, never-flag, the original 3-feature IsolationForest, the current
  ensemble on the old features, and a **VIX rule** (alert when VIX's 52-week z-score
  exceeds a threshold chosen on the validation window). The VIX rule is the bar a
  serious model has to clear.
- **Case studies** in every report: 2008, 2011–12, 2020 week by week.
- Year-block bootstrap kept for uncertainty on every headline number.

### 3.7 Models
Candidates, all on the same features and folds:
1. Regularized logistic regression (strong, interpretable floor)
2. Gradient-boosted trees (`HistGradientBoostingClassifier` or `GradientBoostingClassifier`)
3. The current two-stage design (IsolationForest score → stacking) on v1 features

Selection rule, **fixed before running**: best mean event F1 across development
folds, ties broken by fewer false-alarm episodes per calm year, then by log loss.
The simplest model within one bootstrap standard error of the best wins.

### 3.8 Calibration and threshold
- **Sigmoid (Platt) calibration only.** Two parameters suit validation windows with
  8–10 positives (M7), and it exports to ONNX as a tiny graph (M8).
- Implemented as an explicit `PlattCalibrator(a, b)` fitted on validation scores,
  not `CalibratedClassifierCV(FrozenEstimator(...))`.
- **Threshold per fold, chosen on the validation window:** the lowest threshold whose
  false-alarm rate in calm validation weeks is ≤ the budget (decision T1, default 5%).
  The chosen threshold is stored with the artifact.

### 3.9 Explanations
- Explain the **final calibrated probability**, not a base learner.
- Default: `shap.PermutationExplainer` (model-agnostic) with a 100-row background
  sample from the training window; for tree models, `TreeExplainer` on the margin is
  allowed because Platt calibration is monotonic (state which was used in metadata).
- Output top-k features with readable labels and signed contributions; test that
  contributions plus base value reproduce the model output within tolerance.

### 3.10 Artifacts, ONNX, registry
- `artifacts/<model_version>/`:
  `model.joblib` (scikit-learn, used for SHAP and as parity reference),
  `model.onnx` (scaler + model + Platt graph, or chained stage graphs),
  `metadata.json` (feature spec hash, feature order, label set, horizon, training range,
  data snapshot hash, library versions, calibration params, threshold, metrics),
  `model_card.md`.
- **ONNX parity test** in CI on synthetic data: max abs difference ≤ 1e-5 between
  onnxruntime and scikit-learn probabilities. `skl2onnx` and `onnxruntime` added as
  dev dependencies. A model that fails export cannot be selected.
- **`model_versions` table** (third Alembic migration): version, created_at, label set,
  heads, metrics JSON, artifact path, active flag.
- **`regime_scores` table** (same migration): week, model_version, head, probability,
  alert, top_features JSON.
- CLI `mad-model`: `train`, `backtest`, `export`, `score-history`, `activate`.

### 3.11 Reproducibility and CI
- Full training needs your database (live data), so it runs locally.
- CI runs on synthetic data: feature builder, no-look-ahead, labels and episodes,
  event metrics, calibration, threshold policy, ONNX parity, SHAP additivity.
- `metadata.json` hashes the exact feature frame used, so a run can be matched to
  its inputs.
- Committed outputs: reports in `docs/reports/` (metrics only). Artifacts and feature
  frames stay git-ignored (vendor data).

---

## 4. Steps (one PR each)

**2.0 Prerequisites:** section 2.

**2.1 Feature spec and builder**
Done when: features build from the database for 2000–present; a **no-look-ahead test**
passes (features for week t computed from data truncated at t equal the features from
full data); building twice gives identical output.

**2.2 Labels and episodes**
Done when: `detect`/`predict` targets and episode extraction are tested on synthetic
labels, and the `ews_v1` episode list is written to `docs/reports/episodes-ews_v1.md`.

**2.3 Evaluation harness v2 + baselines**
Done when: the harness runs all baselines (including the current ensemble on old
features and the VIX rule) on development folds and writes
`docs/reports/<date>-baselines.md`. No new model yet; this fixes the bar.

**2.4 Candidate models on development folds**
Done when: all candidates are evaluated with Platt calibration and the threshold policy,
the selection rule picks one, and `docs/reports/<date>-model-selection.md` shows every
candidate, not just the winner. If nothing beats the VIX rule, say so and stop for a
decision.

**2.5 Final confirmation (2018–2021), run once**
Done when: the selected model's confirmation results are reported unedited, with the
2020 case study, in `docs/reports/<date>-final-confirmation.md`.

**2.6 Explanations**
Done when: SHAP on the final probability passes the additivity test and the 2008 and
2020 case studies show top features per week.

**2.7 Artifacts, ONNX, registry**
Done when: `mad-model train` + `export` produce the artifact set, the ONNX parity test
passes in CI, migration 3 adds `model_versions` and `regime_scores`, and the drift
test covers them.

**2.8 Score history and docs**
Done when: `mad-model score-history` writes weekly scores for 2000–present into
`regime_scores`, the model card is written, and `docs/architecture-plan.md` sections 3
and 8 are updated. The old CSV-based `train.py`/`backtest.py` path is kept only as the
"old model" baseline.

---

## 5. Acceptance checks for Phase 2

- [ ] Features come only from live providers in the database; no `bloomberg_ews` inputs
- [ ] No-look-ahead test passes
- [ ] All baselines and candidates reported on development folds, including losers
- [ ] Final confirmation run once and reported unedited
- [ ] Selected model beats the VIX rule and the old ensemble on development event F1,
      or the report says plainly that it doesn't
- [ ] 2008, 2011–12, 2020 case studies in the final report
- [ ] Calibration reported (Brier, log loss, reliability); threshold stored per artifact
- [ ] SHAP explains the final probability; additivity test passes
- [ ] ONNX parity ≤ 1e-5 in CI
- [ ] `model_versions` and `regime_scores` migrated; drift test passes
- [ ] `git status` clean after training, export and scoring

---

## 6. Decisions (defaults unless the user says otherwise)

| # | Decision | Default |
|---|---|---|
| L1 | Labels after 2021-04 (open decision 1) | Train and evaluate on `ews_v1` only; post-2021 weeks are scored but marked "unvalidated". A rule-based label set is a separate later project |
| L2 | 2020 labels interleaved with calm weeks (open decision 13) | Keep as labelled; report 2020 both row-level and with episode merging, and state the effect |
| S1 | Training source | Live providers only (3.1) |
| F1 | Revised series NFCI/STLFSI (open decision 11) | Exclude from v1; test as a v2 experiment with the bias stated |
| H1 | Predict horizon | h = 4 weeks |
| T1 | Alert budget | ≤ 5% false-alarm rate in calm validation weeks |
| C1 | Calibration | Platt only |
| E1 | Development / final split | Develop on 2008–2017; confirm on 2018–2021 once |
| R1 | Redis | Move from Phase 5c to Phase 4 as a shared cache module (no Phase 2 work) |

---

## 7. Risks

- **Few episodes.** About 6–10 labelled episodes in total; event metrics are noisy.
  Always report bootstrap intervals and per-episode results.
- **Overfitting by iteration.** Every look at the development folds costs a little
  honesty; the final set exists for that reason and must not be rerun after tuning.
- **Proxy skew.** Live series differ from Bloomberg (coverage report). Labels were
  made on Bloomberg-era intuition; check that the label/feature relationship holds.
- **Futures rolls and negative prices** (`GC=F`, `CL=F`) create artificial returns;
  validation flags exist from Phase 1 — consider clipping or excluding roll weeks.
- **Adjusted ETF history changes** (v2 only) after dividends; features must be rebuilt
  from current data, not cached.
- **Results may not beat the VIX rule.** That is a valid finding and must be reported.
