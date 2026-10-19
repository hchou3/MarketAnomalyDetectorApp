# Baselines Report — ews_v1 Development Folds (2008–2017)

Generated: 2026-10-19  
Data source: `data/raw/ews_weekly_2000_2021.csv` (EWS Bloomberg labels)  
Label set: `ews_v1` · Horizon h = 4 · Merge gap = 2 · FAR budget = 5%  
Development folds: test years 2008–2017 (expanding window, val ≥ 8 positives)  

> **Note:** The 2018–2021 confirmation window contains only 6 episodes
> (2018-10, 2018-12, 2020-02, 2020-06, 2020-08, 2020-10) and will be run
> once at the end of Phase 2 Step 5. These results are development-only.

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
| Persistence (prev-week label) | 0.868 | 0.847 | 0.857 | 0.040 | 105 | 16 | 19 |
| VIX z-score rule | 0.429 | 0.024 | 0.046 | 0.010 | 3 | 4 | 121 |

---

## Event-level metrics — detect head

### Per-episode hit/miss/lead-time (merge\_gap = 2, VIX rule)

| # | Start | End | Weeks | VIX hit | VIX weeks-in | Persist hit | Always hit |
|---|-------|-----|-------|---------|--------------|-------------|------------|
| 1 | 2008-06-17 | 2008-07-08 | 4 | N | - | Y | Y |
| 2 | 2008-08-26 | 2009-05-05 | 37 | N | - | Y | Y |
| 3 | 2009-06-23 | 2009-06-30 | 2 | N | - | Y | Y |
| 4 | 2010-01-26 | 2010-02-02 | 2 | Y | 0 | Y | Y |
| 5 | 2010-05-04 | 2010-06-29 | 9 | Y | 8 | Y | Y |
| 6 | 2010-08-10 | 2010-08-24 | 3 | N | - | Y | Y |
| 7 | 2011-05-24 | 2011-05-31 | 2 | N | - | Y | Y |
| 8 | 2011-07-19 | 2012-01-31 | 29 | N | - | Y | Y |
| 9 | 2012-04-03 | 2012-04-10 | 2 | N | - | Y | Y |
| 10 | 2012-05-08 | 2012-09-04 | 18 | N | - | Y | Y |
| 11 | 2012-10-30 | 2012-11-06 | 2 | N | - | Y | Y |
| 12 | 2014-09-30 | 2014-10-07 | 2 | N | - | Y | Y |
| 13 | 2015-08-11 | 2015-09-08 | 5 | Y | 2 | Y | Y |
| 14 | 2015-12-29 | 2016-01-26 | 5 | N | - | Y | Y |

**Event recall (merge\_gap=2):** VIX 3/14 (21%), Persistence 14/14 (100%), Always-flag 14/14 (100%)

### Sensitivity to merge gap (VIX rule, detect head)

| Merge gap | Episodes | VIX hits | Recall |
|-----------|----------|----------|--------|
| 0 | 15 | 3 | 0.200 |
| 2 | 14 | 3 | 0.214 |
| 4 | 13 | 3 | 0.231 |

---

## Predict head — calm-at-t analysis

Evaluated on 398 calm-at-t weeks (detect==0 with valid predict label).  
Calm-at-t positives (predict==1 while calm): **57**

### Row-level metrics (calm-at-t weeks only)

| Baseline | Prec | Recall | F1 | FAR |
|----------|------|--------|-----|-----|
| Always-flag | 0.143 | 1.000 | 0.251 | 1.000 |
| Persistence (prev-week label) | 0.125 | 0.035 | 0.055 | 0.041 |
| VIX z-score rule | 0.250 | 0.018 | 0.033 | 0.009 |

### Event-level advance warning (VIX rule, calm-at-t alerts, h=4)

VIX rule: 1/14 (7%) episodes warned  
Persistence: 1/14 (7%) episodes warned  
Always-flag (calm): 14/14 (100%) episodes warned

| # | Episode start | VIX warned | VIX lead (w) | Persist warned | Always warned |
|---|--------------|------------|--------------|----------------|---------------|
| 1 | 2008-06-17 | N | - | N | Y |
| 2 | 2008-08-26 | N | - | N | Y |
| 3 | 2009-06-23 | N | - | N | Y |
| 4 | 2010-01-26 | N | - | N | Y |
| 5 | 2010-05-04 | Y | 1 | N | Y |
| 6 | 2010-08-10 | N | - | N | Y |
| 7 | 2011-05-24 | N | - | N | Y |
| 8 | 2011-07-19 | N | - | N | Y |
| 9 | 2012-04-03 | N | - | N | Y |
| 10 | 2012-05-08 | N | - | Y | Y |
| 11 | 2012-10-30 | N | - | N | Y |
| 12 | 2014-09-30 | N | - | N | Y |
| 13 | 2015-08-11 | N | - | N | Y |
| 14 | 2015-12-29 | N | - | N | Y |

---

## False alarm runs (detect head, calm-year metric)

Grace window: ±2 weeks around each episode. Known-unlabeled windows excluded from count.
Calm dev years (no labeled episode): [2013, 2017]

| Baseline | Total FA runs | FA in calm years | FA runs in known-unlabeled |
|----------|--------------|-----------------|---------------------------|
| VIX rule | 3 | 2 | 0 |
| Persistence | 1 | 0 | 0 |
| Always-flag | — | — | — (all weeks flagged) |

---

## Known events with no EWS label

Alerts in these windows are reported below but excluded from false-alarm counts.

| Event | Window | VIX alerts | Persist alerts |
|-------|--------|------------|----------------|
| 2011-03 Japan/Fukushima turbulence | 2011-02-22 – 2011-04-05 | 0 | 0 |
| 2013-06 Fed taper tantrum | 2013-05-21 – 2013-07-09 | 0 | 0 |
| 2016-06 Brexit vote | 2016-06-14 – 2016-07-19 | 0 | 0 |
| 2018-02 VIX spike (Volmageddon) | 2018-01-30 – 2018-03-13 | 0 | 0 |
| 2019-08 US–China trade-war escalation | 2019-07-30 – 2019-09-10 | 0 | 0 |

---

## Baselines to beat

A candidate model must exceed these bars on the development folds:

- **Event recall (detect, mg=2):** VIX rule = 3/14 (21%)
- **Row-level F1 (detect):** VIX rule = 0.046
- **Calm-year false alarm runs:** VIX rule = 2 in 2 calm years
- **Predict advance-warning recall:** VIX rule = 1/14 (7%)

The VIX rule is the bar. A model that does not beat the VIX rule on event recall is not an improvement over a simple threshold rule.
