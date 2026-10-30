# HGB Experiment Report — ews_v1 (2008–2017)

Generated: 2026-10-29  
Label set: `ews_v1` · FAR budget = 5%  
Dev folds: test years 2008–2017  

> **Pre-registered**: variants specified before seeing results.
> No post-hoc tuning. VIX level is the evaluation bar.
> Hysteresis is serving-logic — not part of the ONNX graph.

---

## Summary table

| Variant | Event F1 | Ep hits | Row F1 | PR AUC | Brier | FA calm | FA total | ONNX |
|---------|----------|---------|--------|--------|-------|---------|----------|------|
| hgb_base | 0.527 | 6/14 | 0.429 | 0.571 | 0.148 | 0 | 6 | pass |
| hgb_vixlvl | 0.627 | 8/14 | 0.470 | 0.607 | 0.141 | 0 | 7 | pass |
| hgb_oof | 0.527 | 6/14 | 0.429 | 0.503 | 0.161 | 0 | 6 | pass |
| hgb_hyst | 0.505 | 6/14 | 0.502 | 0.571 | 0.148 | 0 | 5 | pass |
| hgb_epw | 0.471 | 6/14 | 0.415 | 0.547 | 0.152 | 1 | 10 | pass |
| hgb_all | 0.437 | 5/14 | 0.414 | 0.533 | 0.177 | 0 | 7 | pass |
| vix_level | 0.511 | 6/14 | 0.556 | 0.654 | 0.161 | 0 | 7 | FAIL: VIX rule — no ONNX export |

---

## Bootstrap CIs (year-block, 2000 resamples)

| Variant | Event F1 | 95% CI | Row F1 CI | PR AUC CI | Brier CI |
|---------|----------|--------|-----------|-----------|---------|
| hgb_base | 0.527 | [0.158, 0.724] | [0.128, 0.633] | [0.332, 0.723] | [0.097, 0.200] |
| hgb_vixlvl | 0.627 | [0.397, 0.716] | [0.217, 0.649] | [0.383, 0.752] | [0.092, 0.193] |
| hgb_oof | 0.527 | [0.158, 0.724] | [0.128, 0.633] | [0.289, 0.683] | [0.086, 0.241] |
| hgb_hyst | 0.505 | [0.169, 0.681] | [0.203, 0.683] | [0.332, 0.723] | [0.097, 0.200] |
| hgb_epw | 0.471 | [0.167, 0.623] | [0.122, 0.620] | [0.311, 0.694] | [0.100, 0.204] |
| hgb_all | 0.437 | [0.000, 0.621] | [0.100, 0.611] | [0.324, 0.693] | [0.082, 0.276] |
| vix_level | 0.511 | [0.267, 0.690] | [0.278, 0.697] | [0.380, 0.848] | [0.077, 0.255] |

## Paired bootstrap differences (variant − vix_level)

Positive = variant better than vix_level.

| Variant | ΔEvent F1 | 95% CI | ΔRow F1 | 95% CI |
|---------|-----------|--------|---------|--------|
| hgb_base | -0.006 | [-0.289, 0.231] | -0.135 | [-0.381, 0.048] |
| hgb_vixlvl | +0.100 | [-0.152, 0.310] | -0.087 | [-0.307, 0.069] |
| hgb_oof | -0.006 | [-0.289, 0.231] | -0.135 | [-0.381, 0.048] |
| hgb_hyst | -0.020 | [-0.276, 0.185] | -0.056 | [-0.185, 0.053] |
| hgb_epw | -0.050 | [-0.291, 0.175] | -0.144 | [-0.417, 0.053] |
| hgb_all | -0.089 | [-0.351, 0.145] | -0.147 | [-0.405, 0.015] |

## Paired bootstrap differences (variant − hgb_base)

| Variant | ΔEvent F1 | 95% CI | ΔRow F1 | 95% CI |
|---------|-----------|--------|---------|--------|
| hgb_vixlvl | +0.106 | [-0.025, 0.276] | +0.048 | [-0.010, 0.141] |
| hgb_oof | +0.000 | [0.000, 0.000] | +0.000 | [0.000, 0.000] |
| hgb_hyst | -0.015 | [-0.056, 0.028] | +0.079 | [-0.009, 0.249] |
| hgb_epw | -0.044 | [-0.118, 0.015] | -0.009 | [-0.064, 0.058] |
| hgb_all | -0.083 | [-0.201, 0.000] | -0.012 | [-0.077, 0.070] |

---

## Per-episode hit table

| # | Start | End | Wks | hgb_base | hgb_vixlvl | hgb_oof | hgb_hyst | hgb_epw | hgb_all | vix_level |
|---|-------|-----|-----|---|---|---|---|---|---|---|
| 1 | 2008-06-17 | 2008-07-08 | 4 | Y | Y | Y | Y | Y | Y | N |
| 2 | 2008-08-26 | 2009-05-05 | 37 | Y | Y | Y | Y | Y | Y | Y |
| 3 | 2009-06-23 | 2009-06-30 | 2 | N | N | N | N | N | N | Y |
| 4 | 2010-01-26 | 2010-02-02 | 2 | N | N | N | N | N | N | N |
| 5 | 2010-05-04 | 2010-06-29 | 9 | N | Y | N | N | N | N | Y |
| 6 | 2010-08-10 | 2010-08-24 | 3 | N | N | N | N | N | N | N |
| 7 | 2011-05-24 | 2011-05-31 | 2 | N | N | N | N | N | N | N |
| 8 | 2011-07-19 | 2012-01-31 | 29 | Y | Y | Y | Y | Y | Y | Y |
| 9 | 2012-04-03 | 2012-04-10 | 2 | N | N | N | N | N | N | N |
| 10 | 2012-05-08 | 2012-09-04 | 18 | N | Y | N | N | N | N | N |
| 11 | 2012-10-30 | 2012-11-06 | 2 | N | N | N | N | N | N | N |
| 12 | 2014-09-30 | 2014-10-07 | 2 | Y | Y | Y | Y | Y | N | N |
| 13 | 2015-08-11 | 2015-09-08 | 5 | Y | Y | Y | Y | Y | Y | Y |
| 14 | 2015-12-29 | 2016-01-26 | 5 | Y | Y | Y | Y | Y | Y | Y |

---

## Operating curve at FAR budgets 2%, 5%, 10%

FA/yr denominator: 10 dev years (2008–2017).  
Episodes total: 14.  
Calm years (no episode): [2013, 2017]

| Variant | Budget | Ep caught | FA/yr |
|---------|--------|-----------|-------|
| hgb_base | 2% | 4/14 | 0.80 |
| hgb_base | 5% | 6/14 | 0.60 |
| hgb_base | 10% | 8/14 | 0.80 |
| hgb_vixlvl | 2% | 4/14 | 0.50 |
| hgb_vixlvl | 5% | 8/14 | 0.70 |
| hgb_vixlvl | 10% | 8/14 | 1.20 |
| hgb_oof | 2% | 4/14 | 0.80 |
| hgb_oof | 5% | 6/14 | 0.60 |
| hgb_oof | 10% | 8/14 | 0.80 |
| hgb_hyst | 2% | 4/14 | 0.50 |
| hgb_hyst | 5% | 6/14 | 0.50 |
| hgb_hyst | 10% | 8/14 | 0.60 |
| hgb_epw | 2% | 4/14 | 0.60 |
| hgb_epw | 5% | 6/14 | 1.00 |
| hgb_epw | 10% | 9/14 | 1.10 |
| hgb_all | 2% | 4/14 | 0.50 |
| hgb_all | 5% | 5/14 | 0.70 |
| hgb_all | 10% | 10/14 | 0.80 |
| vix_level | 2% | 5/14 | 0.60 |
| vix_level | 5% | 6/14 | 0.70 |
| vix_level | 10% | 7/14 | 1.10 |
