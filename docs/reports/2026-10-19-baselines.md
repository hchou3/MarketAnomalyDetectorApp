# Baselines Report — ews_v1 Development Folds (2008–2017)

Generated: 2026-10-19  
VIX source: `data/raw/ews_weekly_2000_2021.csv` (CSV reference — prod uses DB)  
Label set: `ews_v1` · Horizon h = 4 · Merge gap = 2 · FAR budget = 5%  
Development folds: test years 2008–2017 (expanding window, val ≥ 8 positives)  

> **Note:** The 2018–2021 confirmation window contains only 6 episodes
> (2018-10, 2018-12, 2020-02, 2020-06, 2020-08, 2020-10) and will be run
> once at the end of Phase 2 Step 5. These results are development-only.

> **Persistence** is an oracle reference baseline: it requires last week's
> label, which is not available at live scoring time. It is reported for
> context but excluded from the model target bar.

---

## Summary

- Out-of-sample dev weeks: 522  
- Anomaly weeks: 124 (23.8%)  
- Episodes (merge\_gap=2, dev): 14  
- Calm-at-t predict positives: 57 (of 398 calm weeks with predict label)  

---

## Row-level metrics — detect head

| Baseline | Prec | Recall | F1 | FAR | TP | FP | FN |
|----------|------|--------|-----|-----|----|----|-----|
| Always-flag | 0.238 | 1.000 | 0.384 | 1.000 | 124 | 398 | 0 |
| Persistence (oracle ref.) *(oracle)* | 0.868 | 0.847 | 0.857 | 0.040 | 105 | 16 | 19 |
| VIX level | 0.794 | 0.435 | 0.562 | 0.035 | 54 | 14 | 70 |
| VIX z52 (log level) | 0.769 | 0.161 | 0.267 | 0.015 | 20 | 6 | 104 |

---

## Event-level metrics — detect head

### Per-episode hit/miss/lead-time (merge\_gap = 2)

| # | Start | End | Wks | Level hit | Level w-in | z52 hit | z52 w-in | Persist† | Always |
|---|-------|-----|-----|-----------|-----------|---------|---------|---------|--------|
| 1 | 2008-06-17 | 2008-07-08 | 4 | N | - | N | - | Y | Y |
| 2 | 2008-08-26 | 2009-05-05 | 37 | Y | 3 | Y | 6 | Y | Y |
| 3 | 2009-06-23 | 2009-06-30 | 2 | Y | 0 | N | - | Y | Y |
| 4 | 2010-01-26 | 2010-02-02 | 2 | N | - | N | - | Y | Y |
| 5 | 2010-05-04 | 2010-06-29 | 9 | Y | 2 | Y | 2 | Y | Y |
| 6 | 2010-08-10 | 2010-08-24 | 3 | N | - | N | - | Y | Y |
| 7 | 2011-05-24 | 2011-05-31 | 2 | N | - | N | - | Y | Y |
| 8 | 2011-07-19 | 2012-01-31 | 29 | Y | 3 | Y | 2 | Y | Y |
| 9 | 2012-04-03 | 2012-04-10 | 2 | N | - | N | - | Y | Y |
| 10 | 2012-05-08 | 2012-09-04 | 18 | N | - | N | - | Y | Y |
| 11 | 2012-10-30 | 2012-11-06 | 2 | N | - | N | - | Y | Y |
| 12 | 2014-09-30 | 2014-10-07 | 2 | N | - | N | - | Y | Y |
| 13 | 2015-08-11 | 2015-09-08 | 5 | Y | 2 | Y | 2 | Y | Y |
| 14 | 2015-12-29 | 2016-01-26 | 5 | N | - | N | - | Y | Y |

† Persistence is an oracle reference (last week's label).

**Event recall (merge\_gap=2):** VIX level 5/14 (35%), VIX z52 4/14 (28%), Persistence† 14/14 (100%), Always-flag 14/14 (100%)

### Sensitivity to merge gap

| Merge gap | Episodes | Level hits | Level recall | z52 hits | z52 recall |
|-----------|----------|-----------|--------------|---------|-----------|
| 0 | 15 | 6 | 0.400 | 5 | 0.333 |
| 2 | 14 | 5 | 0.357 | 4 | 0.286 |
| 4 | 13 | 5 | 0.385 | 4 | 0.308 |

---

## VIX level inside dev episodes

Episodes with median VIX < 20 cannot be caught by any VIX threshold rule (marked **low-VIX**).

| # | Start | End | Wks | VIX median | VIX max | Catchable |
|---|-------|-----|-----|-----------|---------|-----------|
| 1 | 2008-06-17 | 2008-07-08 | 4 | 22.8 | 23.6 | Y |
| 2 | 2008-08-26 | 2009-05-05 | 37 | 44.1 | 67.6 | Y |
| 3 | 2009-06-23 | 2009-06-30 | 2 | 28.5 | 30.6 | Y |
| 4 | 2010-01-26 | 2010-02-02 | 2 | 23.0 | 24.6 | Y |
| 5 | 2010-05-04 | 2010-06-29 | 9 | 33.5 | 35.5 | Y |
| 6 | 2010-08-10 | 2010-08-24 | 3 | 24.3 | 27.5 | Y |
| 7 | 2011-05-24 | 2011-05-31 | 2 | 16.6 | 17.8 | N — low-VIX |
| 8 | 2011-07-19 | 2012-01-31 | 29 | 31.2 | 40.8 | Y |
| 9 | 2012-04-03 | 2012-04-10 | 2 | 18.0 | 20.4 | N — low-VIX |
| 10 | 2012-05-08 | 2012-09-04 | 18 | 18.8 | 24.7 | N — low-VIX |
| 11 | 2012-10-30 | 2012-11-06 | 2 | 17.7 | 17.8 | N — low-VIX |
| 12 | 2014-09-30 | 2014-10-07 | 2 | 16.8 | 17.2 | N — low-VIX |
| 13 | 2015-08-11 | 2015-09-08 | 5 | 24.9 | 36.0 | Y |
| 14 | 2015-12-29 | 2016-01-26 | 5 | 22.5 | 26.1 | Y |

5 of 14 dev episodes have median VIX < 20; no level threshold can catch them.

---

## Predict head — calm-at-t analysis

Evaluated on 398 calm-at-t weeks (detect==0 with valid predict label).  
Calm-at-t positives (predict==1 while calm): **57**

### Row-level metrics (calm-at-t weeks only)

| Baseline | Prec | Recall | F1 | FAR |
|----------|------|--------|-----|-----|
| Always-flag | 0.143 | 1.000 | 0.251 | 1.000 |
| Persistence (oracle ref.) *(oracle)* | 0.125 | 0.035 | 0.055 | 0.041 |
| VIX level | 0.286 | 0.070 | 0.113 | 0.029 |
| VIX z52 (log level) | 0.333 | 0.035 | 0.063 | 0.012 |

### Event-level advance warning (calm-at-t alerts, h=4)

VIX level: 1/14 (7%) episodes warned  
VIX z52: 0/14 (0%) episodes warned  
Persistence†: 1/14 (7%) episodes warned  
Always-flag (calm): 14/14 (100%) episodes warned

| # | Ep start | Level warned | Level lead (w) | z52 warned | z52 lead (w) | Persist† | Always |
|---|----------|-------------|----------------|-----------|-------------|---------|--------|
| 1 | 2008-06-17 | N | - | N | - | N | Y |
| 2 | 2008-08-26 | N | - | N | - | N | Y |
| 3 | 2009-06-23 | Y | 4 | N | - | N | Y |
| 4 | 2010-01-26 | N | - | N | - | N | Y |
| 5 | 2010-05-04 | N | - | N | - | N | Y |
| 6 | 2010-08-10 | N | - | N | - | N | Y |
| 7 | 2011-05-24 | N | - | N | - | N | Y |
| 8 | 2011-07-19 | N | - | N | - | N | Y |
| 9 | 2012-04-03 | N | - | N | - | N | Y |
| 10 | 2012-05-08 | N | - | N | - | Y | Y |
| 11 | 2012-10-30 | N | - | N | - | N | Y |
| 12 | 2014-09-30 | N | - | N | - | N | Y |
| 13 | 2015-08-11 | N | - | N | - | N | Y |
| 14 | 2015-12-29 | N | - | N | - | N | Y |

---

## False alarm runs (detect head, calm-year metric)

Grace window: ±2 weeks around each episode. Known-unlabeled windows excluded from count.
Calm dev years (no labeled episode): [2013, 2017]

| Baseline | Total FA runs | FA in calm years | FA in known-unlabeled |
|----------|--------------|-----------------|----------------------|
| VIX level | 6 | 0 | 0 |
| VIX z52 | 2 | 1 | 0 |
| Persistence† | 1 | — | 0 |
| Always-flag | — | — | — (all weeks flagged) |

---

## Known events with no EWS label

Alerts in these windows are reported below but excluded from false-alarm counts.

| Event | Window | Level alerts | z52 alerts |
|-------|--------|-------------|------------|
| 2011-03 Japan/Fukushima turbulence | 2011-02-22 – 2011-04-05 | 0 | 0 |
| 2013-06 Fed taper tantrum | 2013-05-21 – 2013-07-09 | 0 | 0 |
| 2016-06 Brexit vote | 2016-06-14 – 2016-07-19 | 0 | 0 |
| 2018-02 VIX spike (Volmageddon) | 2018-01-30 – 2018-03-13 | 0 | 0 |
| 2019-08 US–China trade-war escalation | 2019-07-30 – 2019-09-10 | 0 | 0 |

---

## Baselines to beat

A candidate model must exceed **both** VIX level and VIX z52 on the development folds. Persistence is excluded (oracle reference — not achievable live).

- **Event recall (detect, mg=2):** VIX level 5/14 (35%), VIX z52 4/14 (28%) → beat 5/14 (35%)
- **Row-level F1 (detect):** VIX level 0.562, VIX z52 0.267 → beat 0.562
- **Calm-year false alarm runs:** VIX level 0, VIX z52 1 in 2 calm years
- **Predict advance-warning recall:** VIX level 1/14 (7%), VIX z52 0/14 (0%) → beat 1/14 (7%)
