# Walk-Forward Backtest — 2008 to 2021

**Run:** originally 2026-10-10 on `walk-forward-backtest`; reproduced exactly on
2026-10-13 during Phase 0 (identical `summary.json` sections and byte-identical
`predictions.csv`).
**Command:** `mad-backtest` (default settings).
**Versions:** Python 3.13, numpy 2.2.6, pandas 2.3.1, scikit-learn 1.6.1, shap 0.50.0
(the pinned versions in `uv.lock`). Results change under other library versions.

The raw outputs (`predictions.csv`, `per_year.csv`, `summary.json`) are generated in
`artifacts/backtest/` and are not tracked in git. This report keeps the findings.

## Method

- One fold per calendar test year, 2008 to 2021-04 (14 folds, 695 out-of-sample weeks,
  153 anomaly weeks, prevalence 22.0%).
- Training uses only weeks before the calibration window. The calibration window is
  the two calendar years before the test year, extended backwards one year at a time
  until it holds at least 8 anomaly weeks and 8 normal weeks. Only past labels decide
  the extension.
- The model and features are unchanged from `train.py`: the two-stage ensemble on the
  8 raw-level features, calibration chosen per fold by out-of-fold Brier score.
- Baselines: the standalone 3-feature Isolation Forest from the original notebook, and
  "always flag anomaly."
- Decision threshold 0.5.

## Pooled results

| Model | Precision | Recall | F1 | False-alarm rate | Caught | False alarms |
|---|---|---|---|---|---|---|
| Ensemble | 0.723 | 0.562 | 0.632 | 6.1% | 86 / 153 | 33 |
| Baseline Isolation Forest | 0.324 | 0.752 | 0.453 | 44.3% | 115 / 153 | 240 |
| Always flag anomaly | 0.220 | 1.000 | 0.361 | 100% | 153 / 153 | 542 |

Threshold-free: ensemble PR AUC 0.714 vs. baseline 0.615; ROC AUC 0.880 vs. 0.768.
Ensemble Brier score 0.103.

**Uncertainty.** A bootstrap over calendar years (2,000 resamples) gives a pooled F1
difference (ensemble − baseline) 95% CI of [−0.098, +0.415], with the ensemble ahead in
91.5% of resamples. The improvement is likely but not established at conventional
levels. Per-year F1, over the 10 years containing anomalies, averages 0.447 for both
models (ensemble better in 6 years, baseline in 3, tied in 1).

## Per year

| Year | Anomaly weeks | Ensemble caught | Missed | False alarms | Baseline caught | Baseline false alarms | Calibration | Val window |
|---|---|---|---|---|---|---|---|---|
| 2008 | 27 | **0** | 27 | 0 | 27 | 26 | isotonic | 2006–2007 |
| 2009 | 20 | 17 | 3 | 0 | 20 | 31 | sigmoid | 2007–2008 |
| 2010 | 14 | 7 | 7 | 2 | 14 | 24 | sigmoid | 2008–2009 |
| 2011 | 24 | 23 | 1 | 6 | 19 | 2 | sigmoid | 2009–2010 |
| 2012 | 27 | 15 | 12 | 0 | 11 | 2 | sigmoid | 2010–2011 |
| 2013 | 0 | 0 | 0 | 0 | 0 | 36 | sigmoid | 2011–2012 |
| 2014 | 2 | 0 | 2 | 0 | 0 | 22 | sigmoid | 2012–2013 |
| 2015 | 6 | 2 | 4 | 0 | 5 | 21 | sigmoid | 2012–2014 |
| 2016 | 4 | 0 | 4 | 0 | 1 | 7 | sigmoid | 2014–2015 |
| 2017 | 0 | 0 | 0 | 0 | 0 | 43 | sigmoid | 2015–2016 |
| 2018 | 6 | 1 | 5 | 2 | 1 | 6 | sigmoid | 2015–2017 |
| 2019 | 0 | 0 | 0 | 1 | 0 | 1 | isotonic | 2016–2018 |
| 2020 | 23 | 21 | 2 | **22** | 17 | 11 | isotonic | 2016–2019 |
| 2021 (to Apr 20) | 0 | 0 | 0 | 0 | 0 | 8 | sigmoid | 2019–2020 |

## Findings for Phase 2

1. **2008 was missed entirely.** The 2008 fold trains on 2000–2005 and calibrates on
   2006–2007, a window with only 8 anomaly weeks, and the model never crosses 0.5.
   The raw-level features (gold and DXY levels, `USD_Over_100`) describe the
   2000–2005 regime, so a crisis at different price levels looks unfamiliar in the
   wrong way. Stationary features are the main fix.
2. **2020 produced 22 false alarms** alongside 21 correct detections. The 2020 labels
   are not one block: the 23 anomaly weeks run from Feb 11 to Nov 3 with calm weeks
   between them. The false alarms fall in those calm gaps (16 between May 19 and
   Oct 13) and after the last labelled week (6 from Nov 10 to Dec 29). The model
   treats mid-2020 as one continuous stress period; whether that's a model problem or
   a labelling choice is worth deciding before tuning against it.
3. **Small-episode years are mostly missed** (2014–2018: 3 of 18 anomaly weeks caught),
   though with very few false alarms. Event-level metrics (was each episode flagged at
   all, and how early) will describe this better than week counts.
4. **Calibration windows are thin.** From 2016 on, the window often has the minimum
   8–10 anomaly weeks, and isotonic was selected three times. With so few positives,
   isotonic can overfit; Phase 2 should compare against sigmoid-only and report log
   loss alongside Brier.
