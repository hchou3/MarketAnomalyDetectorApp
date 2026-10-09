# Instruction: Build Phase 1 of the Market Anomaly Detector

You are picking up the `market-anomaly-detector` project. Phase 1 goal (from the project spec): **get the ensemble working locally, no infra** — a single `score(df)` function that runs a real two-stage anomaly pipeline with a calibrated confidence score and per-prediction SHAP explanations.

## Source material

The existing notebook (`Market_Anomaly_Detector.ipynb`) is exploratory, not production code. Reuse its useful pieces, but do not port it wholesale. Specifically:

**Reuse:**
- Data file: `FinancialMarketData.xlsx - EWS.csv` — weekly macro/ticker time series with a `Data` date column and binary `Y` anomaly target.
- The feature engineering logic (rolling averages, MA-cross signals, `USD_Over_100` flag), but rewritten as a single clean function, not copy-pasted cells.
- The time-based train/val/test split logic (train 9y / val 2y / test 1y, sorted by date, no shuffling) — this is correct for time series and should stay.
- The four model types already explored: Logistic Regression, Decision Tree, Random Forest, Isolation Forest.

**Fix (these are bugs/gaps in the notebook, not design choices to preserve):**
- Each model was trained on a *different* ad hoc feature subset (LR/DT/RF used 8 engineered features; Isolation Forest used only 3). Phase 1 needs one consistent feature set feeding the whole pipeline.
- The model-saving cells pickle the filename string instead of the fitted model (e.g. `pickle.dump("iso_model.pkl", file)` instead of `pickle.dump(iso_model, file)`). Don't repeat this.
- No model currently produces a probability/confidence — only hard `.predict()` labels. Fix per the requirements below.
- SHAP is computed once at training time, not per prediction. Fix per the requirements below.
- There is no combination logic between the four models at all today — that's the core gap Phase 1 closes.

**Do not carry over** (out of scope for Phase 1, belongs to later phases or nowhere):
- Colab-specific code: `google.colab.userdata`, API key prompts, `!pip install` cells.
- The synthetic data generator (`np.random.uniform` fake ticker data standing in for live data).
- The Gemini/OpenAI/Groq "investment strategy" text generation.
- The Streamlit app + ngrok tunnel — that's Phase 2 (real API), not Phase 1.
- SMOTE — the notebook's own comment notes it made results worse; the final training path already dropped it in favor of `class_weight='balanced'`. Keep it dropped.

## What to build

Create a proper local package (not a notebook) with roughly this shape:

```
market_anomaly_detector/
  data.py        # load CSV, feature engineering (rolling avgs, crosses, flags)
  splits.py       # time-based train/val/test split
  pipeline.py     # the two-stage ensemble + calibration + score()
  explain.py       # SHAP wrapper, computed per-prediction
  train.py        # script: trains + saves all artifacts
  tests/           # basic tests, incl. a train/test leakage check
```

### 1. Ensemble architecture
Implement the two-stage cascade described in the spec:
- **Stage 1 — Isolation Forest**: unsupervised filter over the full common feature set, producing an anomaly score (not just a binary flag) that's fed forward as a feature.
- **Stage 2 — stacked classifier**: Decision Tree + Random Forest + Logistic Regression as base learners, combined via `sklearn.ensemble.StackingClassifier` with a simple meta-learner (logistic regression is fine — don't over-engineer this in Phase 1). Include the Isolation Forest score as one of the input features to the stack, so Stage 1's output actually informs Stage 2 rather than being a separate, disconnected model.
- All stages should train and predict on the **same** engineered feature set — resolve the notebook's inconsistent per-model feature subsets by picking one shared set (start from the union of what the notebook's permutation-importance and correlation analysis flagged as useful: `DXY`, `JPY`, `VIX`, `GTITL2YR`, `XAU BGNL`, plus their engineered cross/flag features).

### 2. Confidence score
- Use `predict_proba` from the final stacked model, not a hard label.
- Wrap it in `CalibratedClassifierCV` (sigmoid or isotonic — try both on the validation split, keep whichever calibrates better) so the returned probability is meaningful, not just a raw score.

### 3. Per-prediction SHAP
- Build the SHAP explainer once at load/train time (expensive setup), but call `.shap_values()` / the explainer per incoming row at inference time — not just once globally at training.
- Return the top contributing features per prediction, not just a training-time summary plot.

### 4. Public interface
Wrap everything into one function:

```python
def score(df: pd.DataFrame) -> dict:
    """
    Returns:
      {
        "anomaly": bool,
        "confidence": float,       # calibrated probability
        "top_features": [ {"feature": str, "impact": float}, ... ]
      }
    """
```

It should accept a dataframe in the same raw shape as the source CSV (or a single row/window of it), run feature engineering internally, and return the dict above. This is the function Phase 2's FastAPI layer will call directly, so keep the interface stable and don't leak model internals (sklearn objects, raw SHAP arrays) through it.

## Validation

- Reuse the notebook's existing time-based split (9y/2y/1y) — do not random-split time series data.
- Report F1/precision/recall for the new stacked ensemble on the same val/test split, and compare directly against the notebook's original best single model (Isolation Forest at 84–87% F1, per the project's stated baseline). This comparison is the headline metric for the project, so don't skip it.
- Add a basic test that asserts no future data leaks into training (max date in train < min date in val < min date in test).

## Explicitly out of scope right now
FastAPI, Redis, Docker, Kafka, Feast, Triton, gRPC, FAISS, Kubernetes — all later phases. If you find yourself reaching for any of these while doing Phase 1, stop; it belongs later.
