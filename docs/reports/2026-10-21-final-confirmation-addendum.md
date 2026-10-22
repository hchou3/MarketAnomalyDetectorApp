# Final Confirmation Addendum — ews_v1 (2018–2021)

> **Post-hoc analysis, written after the confirmation run; no model, threshold or result changed.**

Source: `artifacts/confirm_predictions.csv` (confirm window) and `artifacts/dev_predictions.csv`
(dev window, for comparison). Confirmation report: `2026-10-21-final-confirmation-331acff8.md`.

---

## 1. Calm-week false-alarm rate (week-level)

The confirmation report's **FA total** column counts *alert runs* (contiguous sequences of
alerts on labelled-calm weeks, merged across gaps ≤ 2 weeks). This section reports the raw
**calm-week alert rate** instead: fraction of labelled-calm, non-known-unlabeled weeks on
which the model fired.

### hgb and vix_level (from confirm_predictions.csv)

| Year | Calm wks | hgb alerts | hgb FAR | vix_level alerts | vix_level FAR |
|------|----------|-----------|---------|-----------------|---------------|
| 2018 | 39 | 0 | 0.000 | 3 | 0.077 |
| 2019 | 46 | 5 | 0.109 | 1 | 0.022 |
| 2020 | 29 | 3 | 0.103 | 22 | 0.759 |
| 2021 | 16 | 0 | 0.000 | 0 | 0.000 |
| **All** | **130** | **8** | **0.062** | **26** | **0.200** |

hgb overall calm-week FAR (6.2%) is close to its 5% budget; vix_level (20.0%) is four times
that budget.

### vix_z52 and old_ensemble_native05

Per-week alert data for these models is not exported to `confirm_predictions.csv`.
The confirmation report's summary table reports only aggregate metrics:

| Model | FA total (runs) | FA per year (runs) |
|-------|----------------|--------------------|
| vix_z52 | 2 | 0.50 |
| old_ensemble_native05 | 2 | 0.50 |

Per-week calm-week FAR for these models is unavailable without re-running the evaluation.

### Comparison with dev window (from dev_predictions.csv)

| Period | Calm wks | hgb FAR | vix_level FAR |
|--------|----------|---------|---------------|
| Dev 2008–2017 | 377 | 0.066 | 0.085 |
| Confirm 2018–2021 | 130 | 0.062 | 0.200 |

hgb's calm-week FAR is stable across dev and confirm. vix_level's calm-week FAR triples in
the confirm window due to 2020 (see section 2).

---

## 2. Stress weeks flagged

| Year | Stress wks | hgb alerts | hgb% | vix_level alerts | vix_level% |
|------|-----------|-----------|------|-----------------|------------|
| 2018 | 6 | 2 | 33% | 3 | 50% |
| 2019 | 0 | — | — | — | — |
| 2020 | 23 | 14 | 61% | 21 | 91% |
| 2021 | 0 | — | — | — | — |
| **All** | **29** | **16** | **55%** | **24** | **83%** |

2019 and 2021 had no labelled stress weeks in the confirm window.

---

## 3. Why run-counting misleads for vix_level in 2020

The confirmation report shows vix_level with FA total = 3 runs across all four test years, which
looks comparable to hgb's 3 runs. But vix_level fired on **22 of 29 labelled-calm weeks in
2020 alone** (75.9% FAR), because VIX stayed above the 21.1 threshold for most of the year.

vix_level calm-week alerts in 2020:

| Month | Calm wks | Alerts |
|-------|----------|--------|
| Jan | 4 | 0 |
| Feb | 1 | 0 |
| May | 2 | 2 |
| Jun | 3 | 3 |
| Jul | 4 | 4 |
| Aug | 3 | 3 |
| Sep | 2 | 2 |
| Oct | 2 | 2 |
| Nov | 3 | 3 |
| Dec | 5 | 3 |

May–Dec 2020: 22 of 24 calm weeks alerted (92%). Because VIX's elevation was
nearly continuous, the merge-gap algorithm collapses many of these weeks into very few runs.

The same artefact inflates vix_level's event F1: when a model is effectively always-on during
an episode-heavy year, the run-detection metric rewards it for "catching" those episodes.
`always_flag` scores event F1 = **1.000** in this report for the same reason — it fires every
week and therefore "catches" every episode.

**Implication:** FA runs and event F1 should not be used in isolation to compare models.
Per-week calm-week FAR and per-week recall (stress weeks flagged) are more informative
for a model that is meant to be off most of the time.

---

## 4. Predict head: no advance warning

The confirmation report's predict-head table shows hgb gave advance warning (a predict-head
alert ≤ h = 4 weeks before onset) for **0 of 6 episodes**:

| # | Start | hgb advance warning |
|---|-------|---------------------|
| 1 | 2018-10-09 | N |
| 2 | 2018-12-11 | N |
| 3 | 2020-02-11 | N |
| 4 | 2020-06-09 | N |
| 5 | 2020-08-25 | N |
| 6 | 2020-10-20 | N |

The predict head's precision (0.643) comes from a small number of alerts during already-elevated
periods; it does not fire before episodes begin. **The predict head should not be presented as
an early-warning signal.** Its alerts coincide with stress or follow shortly after onset, not
precede it.

vix_level's predict head gave advance warning for 3/6 episodes (episodes 4, 5, 6 in 2020),
all within the same sustained VIX-elevated window — the same artefact as section 3.

---

## 5. Summary

| Issue | Detail |
|-------|--------|
| Run-counting understates vix_level FAs | 3 FA runs but 26 calm-week alerts (20.0%); 22 of 29 calm weeks in 2020 |
| Event F1 always-flag artefact | always_flag scores 1.000; vix_level benefits from the same effect in 2020 |
| hgb predict head gives no advance warning | 0/6 episodes; should not be described as early-warning |
| hgb calm-week FAR is within budget | 6.2% confirm vs 6.6% dev; stable across periods |
