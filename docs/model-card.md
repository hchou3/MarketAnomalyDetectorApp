# Model Card — Market Anomaly Detector v1

**Version:** `v1-2026-11-16-b8c2d65c`  
**Created:** 2026-11-16  
**Label set:** `ews_v1`  
**Feature spec:** SPEC_V1 (`b8c2d65c…`, 51 features)

> **Informational only.** This model is a research tool for detecting historical
> market-stress regimes. Its output is not investment advice, a recommendation to
> buy, sell, or hold any security, or a prediction of future market performance.
> Past regime detection does not imply future accuracy. Use of this model for
> trading, risk management, or any financial decision is entirely at the user's
> own risk.

---

## 1. Purpose and scope

The model detects **market-wide stress regimes** — periods when multiple macro
instruments simultaneously signal elevated risk. It outputs a calibrated probability
(0–1) and an alert flag when that probability exceeds a 5% false-alarm-rate threshold
calibrated on a held-out validation window.

The model is market-level only. It does not score individual tickers, sectors, or
portfolios. All instrument descriptions reflect historical co-movement during past
episodes ("gold rose in 4 of 6 past episodes"), never investment advice.

Two heads are produced for every scored week:
- **detect** (head = "serving"): probability that the current week is an anomaly.
  This head fires alerts and is the production signal.
- **predict** (head = "experimental"): probability that an anomaly will begin within
  the next 4 weeks. In the 2018–2021 confirmation window, this head produced 0 of 6
  advance warnings (VIX level produced 3 of 6). It is **never used for alerts**.

---

## 2. Data

### 2.1 Providers and instruments

All features come from live data providers. Bloomberg data is used only for the
`ews_v1` labels and as a fidelity reference; it is not an input to v1 scoring.

| Instrument | Canonical ID | Primary provider | Secondary |
|------------|-------------|-----------------|-----------|
| VIX | `vix` | Cboe | Yahoo `^VIX` |
| S&P 500 | `msci_usa` | Yahoo `^GSPC` | — |
| Dollar Index | `dollar_index` | Yahoo `DX-Y.NYB` | — |
| USD/JPY | `usd_jpy` | Yahoo `JPY=X` | FRED |
| GBP/USD | `gbp_usd` | Yahoo `GBPUSD=X` | FRED |
| Gold | `gold` | Yahoo `GC=F` (front month) | — |
| WTI crude | `wti_front` | Yahoo `CL=F` (front month) | — |
| US 10Y yield | `us_10y_yield` | FRED `DGS10` | — |
| US 2Y yield | `us_2y_yield` | FRED `DGS2` | — |
| US 3M yield | `us_3m_yield` | FRED `DTB3` | — |
| CBOE SKEW | `skew` | Cboe | Yahoo `^SKEW` |

The USD/JPY correlation to Bloomberg (`JPY BGN Curncy`) is ~0.88 on weekly changes;
the timing difference (Yahoo FX closes vs Bloomberg) accounts for most of the
divergence. This is flagged in the Phase 1 coverage report.

### 2.2 Point-in-time rule

For week t (a Tuesday):
- Each daily series uses the last value whose `known_at` ≤ t.
- Weekly and monthly series use the latest published value whose `known_at` ≤ t.
- Staleness caps apply: a series unused for N trading days is flagged stale.
- No forward-fill across more than one period; missing values propagate as NaN.
- Incomplete bars (the in-progress week) are excluded: scoring uses only closed weeks.

This prevents look-ahead: features built as-of week t use only data that was
available on that Tuesday in real time.

---

## 3. Labels

**Label set:** `ews_v1`  
**Source:** Bloomberg EWS CSV, column `Y`  
**Range:** 2000-01-04 – 2021-04-20  
**Labeling rule:** undocumented; known to be derived from Bloomberg market-stress
indicators. The rule itself cannot be recovered from the CSV; this is a known limitation.

The label set ends in April 2021. All weeks after that date are scored with
"unvalidated" status — no labels exist to measure performance there.

### 3.1 Known unlabeled events

The following periods are excluded from false-alarm calculations because they may
represent genuine stress that the labels missed:

| Event | Window |
|-------|--------|
| 2018-02 VIX spike (Volmageddon) | 2018-01-30 – 2018-03-13 |
| 2019-08 US–China trade-war escalation | 2019-07-30 – 2019-09-10 |
| 2016-06 Brexit | 2016-06-07 – 2016-07-12 |
| 2013-06 Taper tantrum | 2013-05-21 – 2013-07-16 |
| 2011-03 Japan earthquake | 2011-03-08 – 2011-04-26 |
| 2019-08-27 | 2019-08-20 – 2019-09-03 |

### 3.2 2020 interleaving

The 2020 anomaly weeks are interleaved with calm weeks (February–November). This
reflects the Bloomberg labeling rule, not an error. The model does not attempt to
smooth these labels; hysteresis is deferred to Phase 4 (serving logic).

---

## 4. Features

**Spec:** SPEC_V1  
**Spec hash (SHA-256):** `b8c2d65c449392a806bf30c8fc596785cab943aa08268fae4fff593b14ab8a3f`  
**Number of features:** 51  
**Anchor:** W-TUE (weekly close, Tuesday)

No raw price levels are used. All features are stationary transforms:

| Transform | Applied to |
|-----------|-----------|
| Log return 1 / 4 / 12 weeks | Prices and indices |
| Drawdown from trailing 52-week high | Prices and indices |
| Realised volatility (20 trading days, annualised) | Prices and indices |
| Yield change in bp 1 / 4 / 12 weeks | Percent series (yields, rates) |
| Spread: 10y−3m, 10y−2y in bp | US yields |
| Z-score vs trailing 156-week window (min 104 weeks) | All |

Spreads and z-scores are the only level-like inputs.

---

## 5. Model

**Type:** `HistGradientBoostingClassifier` (HGB)  
**Hyperparameters:** fixed (not tuned on confirmation data)

| Parameter | Value |
|-----------|-------|
| `max_iter` | 300 |
| `learning_rate` | 0.05 |
| `max_depth` | 4 |
| `min_samples_leaf` | 20 |
| `l2_regularization` | 1.0 |
| `max_features` | 0.8 |
| `random_state` | 42 |

**Calibration:** Platt sigmoid `P = σ(a·z + b)` fitted by NLL minimisation on the
validation window (the held-out 8+ positive weeks immediately before the test period).
Parameters stored per artifact in `metadata.json`.

**Threshold:** 5% false-alarm rate on calm validation weeks. The alert fires when
calibrated probability ≥ stored threshold.

**Selection-rule override:** The automated rule (simplest model within 1 bootstrap SE
of the best dev event F1) selected `logreg`. HGB was substituted because logistic
regression breaks in out-of-distribution regimes: in the 2009 and 2010 development
folds, the Platt calibrator fits a negative slope (`a < 0`) due to linear
extrapolation into the unseen 2008 crisis, producing inverted scores. As a result,
logreg fired an alert in only 3 of 37 weeks of the main 2008 episode, versus HGB's 22.
HGB is better on every secondary metric. This override is documented in
`docs/reports/2026-10-21-model-selection.md`.

**Final fit windows:**
- Detect head: all 2000–2021 labeled data, retrained on the full window after
  the confirmation run. Platt calibration is re-fitted on the final validation window
  (2019-04-23 – 2021-04-20).
- Predict head: same pipeline and raw margin; separate Platt calibration fitted on
  h=4 predict targets.

---

## 6. Results

### 6.1 Development folds (2008–2017)

Walk-forward expanding window, 10 annual test folds, val ≥ 8 positives each.
Full report: `docs/reports/2026-10-21-model-selection.md`.

| Model | Event F1 | Episodes caught | Row F1 | PR AUC | Brier | FA calm runs |
|-------|---------|----------------|--------|--------|-------|--------------|
| **hgb (selected)** | **0.527** | **6/14** | **0.429** | **0.571** | **0.148** | **0** |
| VIX level | 0.511 | 6/14 | 0.556 | 0.654 | 0.161 | 0 |
| logreg | 0.515 | 6/14 | 0.257 | 0.366 | 0.192 | 1 |
| Old ensemble | 0.494 | 5/14 | 0.612 | 0.815 | 0.088 | 0 |

Event F1 = harmonic mean of episode recall and calm-run precision
(merge gap = 2 calm weeks, grace ± 2 weeks).
HGB and VIX level are statistically indistinguishable on dev event F1 (CI straddles zero).

### 6.2 Confirmation (2018–2021) — run once, unedited

6 episodes in the confirmation window.
Full report: `docs/reports/2026-10-21-final-confirmation-331acff8.md`.

| Model | Event F1 | Episodes caught | Row F1 | PR AUC | Brier | Log loss | FA/yr |
|-------|---------|----------------|--------|--------|-------|----------|-------|
| **hgb** | **0.645** | **4/6** | **0.604** | **0.612** | **0.101** | **0.408** | **0.75** |
| VIX level | 0.625 | 5/6 | 0.593 | 0.766 | 0.175 | 0.957 | 0.75 |

HGB missed episodes 1 (2018-10 Q4 onset, 3-week episode) and 5 (2020-08 COVID-19
second wave). VIX level caught 5 of 6. HGB had materially better calibration
(Brier 0.101 vs 0.175) and lower false alarms vs the calm baseline.

**Predict head:** 0/6 advance warnings in the confirmation window (VIX level: 3/6).
The predict head is logged but must not be used for alerts.

---

## 7. Known limitations

1. **Mostly S&P-driven.** The 51 SPEC_V1 features are dominated by S&P 500
   derivatives. Episodes driven by fixed income or currency stress without equity
   contagion may be under-detected.

2. **Rising-rate regimes.** The 3-month T-bill feature captures rate change, not
   level. The 2018 Q4 sell-off (rate hikes + equity decline) was missed in the
   first sub-episode but caught in December. Rate-driven regimes with low VIX
   may fall below the threshold.

3. **Yahoo FX timing.** USD/JPY and GBP/USD from Yahoo have ~0.88 weekly
   correlation with Bloomberg. Any regime primarily driven by FX moves may be
   detected with slightly lower probability than a Bloomberg-sourced model would give.

4. **Event F1 counts runs, not weeks.** A 2-week episode and a 37-week episode
   contribute equally to event recall. This is intentional (early warning matters),
   but it makes the metric sensitive to how episodes are merged.

5. **Labels end 2021-04-20.** Post-2021 scores have no labels for validation.
   They are stored with `status="ok"` and `status="experimental"` but should be
   treated with caution (Decision L1 from architecture plan).

6. **No hysteresis.** Alert = raw threshold crossing. Persistent regime entry/exit
   logic is deferred to Phase 4.

---

## 8. Serving

### 8.1 ONNX export

**Input:** `float64` tensor `[N, 51]`, columns in `feature_order` from `metadata.json`  
**Output:** `float64` tensor `calibrated_probability [N]`  
**Parity tolerance:** ≤ 1e-5 (max absolute difference vs sklearn on training rows)  
**Verified parity:** 3.3e-9 on the production model

Triton configuration requirements:
- Input/output dtype must be `TYPE_FP64` (not `TYPE_FP32`).
- Use `onnxruntime` with `ai.onnx.ml` opset ≥ 3 (`nodes_values_as_tensor` support).
  Verified on onnxruntime 1.31. Earlier versions silently coerce tree thresholds
  to float32 and cause up to 0.05 probability error on bp-change grid inputs.

### 8.2 Parity verification

The scoring path (`run_score_history`, Phase 4 scorer) re-runs the ONNX parity
check on every batch before writing results. If `max_diff > 1e-5`, scoring aborts
and returns a non-zero exit code.

---

## 9. Reproduce

```bash
# 1. Schema
uv run alembic upgrade head

# 2. Backfill data (run on a machine with internet access)
uv run --env-file .env mad-data backfill

# 3. Train
uv run --env-file .env mad-model train --final --label-set ews_v1 --database-url $MAD_DATABASE_URL

# 4. Activate (replace version with the printed version string)
uv run mad-model activate v1-2026-11-16-b8c2d65c --head detect --database-url $MAD_DATABASE_URL

# 5. Score history
uv run mad-model score-history --database-url $MAD_DATABASE_URL

# Verify ONNX parity in CI
pytest tests/test_final_train.py::test_onnx_parity_bp_grid_fixture -v
```

**Pinned versions:** `uv.lock` locks all transitive deps.
Key versions: numpy 2.2.6, scikit-learn 1.6.1, shap 0.50.0, onnxruntime 1.21.0,
skl2onnx 1.20.

**Spec and config hashes:**  
- `spec_hash`: `b8c2d65c449392a806bf30c8fc596785cab943aa08268fae4fff593b14ab8a3f`  
- `config_hash`: `331acff81da50cf5180a53b490b10764daafde0522aaa6a5bda02d0e45ad2c01`
