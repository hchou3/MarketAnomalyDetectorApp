"""Generate the baselines report for docs/reports/<date>-baselines.md."""

from __future__ import annotations

import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from market_anomaly_detector.episodes import extract_episodes
from market_anomaly_detector.eval_harness import (
    KNOWN_UNLABELED,
    always_flag,
    detect_event_results,
    false_alarm_runs,
    fit_vix_threshold,
    make_folds_v2,
    persistence,
    predict_event_results,
    row_metrics,
    vix_rule,
)
from market_anomaly_detector.labels import build_targets

# ── Load EWS data ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/raw/ews_weekly_2000_2021.csv", parse_dates=["Data"])
df = df.sort_values("Data").reset_index(drop=True)
weeks: list[date] = [d.date() for d in df["Data"]]
detect = df["Y"].values.astype(int)

# VIX 156-week z-score of 1-week log return
log_vix = np.log(df["VIX"].values)
vix_lr = np.empty_like(log_vix)
vix_lr[0] = np.nan
vix_lr[1:] = np.diff(log_vix)
vix_z = np.full(len(vix_lr), np.nan)
window, min_p = 156, 104
for i in range(len(vix_lr)):
    w = vix_lr[max(0, i - window + 1) : i + 1]
    w = w[~np.isnan(w)]
    if len(w) >= min_p and not np.isnan(vix_lr[i]):
        std = w.std(ddof=1)
        vix_z[i] = (vix_lr[i] - w.mean()) / std if std > 0 else 0.0

labels_df = pd.DataFrame({"week": weeks, "value": detect})
targets = build_targets(labels_df, horizon=4)
predict_labels = targets["predict"].values  # float, NaN for last 4 rows

ep2 = extract_episodes(labels_df, merge_gap=2)
ep0 = extract_episodes(labels_df, merge_gap=0)
ep4 = extract_episodes(labels_df, merge_gap=4)

# Known-unlabeled mask
ku_mask = np.zeros(len(weeks), dtype=bool)
for ks, ke, _ in KNOWN_UNLABELED:
    for i, w in enumerate(weeks):
        if ks <= w <= ke:
            ku_mask[i] = True

# ── Build folds (dev: 2008–2017) ──────────────────────────────────────────────
folds = make_folds_v2(weeks, detect, first_test_year=2008, last_test_year=2017)
print(f"{len(folds)} development folds")

# ── Walk-forward baseline evaluation ─────────────────────────────────────────
rows_detect: list[dict] = []
rows_predict: list[dict] = []

for fold in folds:
    val_detect = detect[fold.val_lo : fold.test_lo]
    val_vix_z = vix_z[fold.val_lo : fold.test_lo]
    val_ku = ku_mask[fold.val_lo : fold.test_lo]
    thr = fit_vix_threshold(val_vix_z, val_detect, far_budget=0.05, excluded_mask=val_ku)

    test_slice = slice(fold.test_lo, fold.test_hi)
    t_detect = detect[test_slice]
    t_vix_z = vix_z[test_slice]
    t_pred_labels = predict_labels[test_slice]
    t_weeks = weeks[fold.test_lo : fold.test_hi]

    af = always_flag(len(t_detect))
    per = persistence(t_detect)
    vr = vix_rule(t_vix_z, thr)

    for k, (w, dl, pl) in enumerate(zip(t_weeks, t_detect, t_pred_labels)):
        rows_detect.append(
            {
                "week": w,
                "detect": int(dl),
                "always": 1,
                "persist": int(per[k]),
                "vix": int(vr[k]),
                "vix_score": float(t_vix_z[k]) if not np.isnan(t_vix_z[k]) else float("nan"),
                "fold": fold.test_year,
            }
        )
        if not np.isnan(pl):
            rows_predict.append(
                {
                    "week": w,
                    "predict": int(pl),
                    "detect": int(dl),
                    "always": 1,
                    "persist": int(per[k]),
                    "vix": int(vr[k]),
                    "fold": fold.test_year,
                }
            )

det_df = pd.DataFrame(rows_detect)
pred_df = pd.DataFrame(rows_predict)

calm_pred_df = pred_df[pred_df["detect"] == 0].copy()
calm_at_t_pos = int((calm_pred_df["predict"] == 1).sum())
print(
    f"Calm-at-t predict positives: {calm_at_t_pos}"
    f" (of {len(calm_pred_df)} calm weeks with predict label)"
)

# ── Row-level metrics ─────────────────────────────────────────────────────────
baselines = ["always", "persist", "vix"]
baseline_names = {
    "always": "Always-flag",
    "persist": "Persistence (prev-week label)",
    "vix": "VIX z-score rule",
}
detect_metrics = {b: row_metrics(det_df["detect"], det_df[b]) for b in baselines}
predict_metrics = {b: row_metrics(calm_pred_df["predict"], calm_pred_df[b]) for b in baselines}

# ── Event-level metrics ───────────────────────────────────────────────────────
dev_start, dev_end = date(2008, 1, 1), date(2017, 12, 31)
ep2_dev = [e for e in ep2 if dev_start <= e.start <= dev_end]
ep0_dev = [e for e in ep0 if dev_start <= e.start <= dev_end]
ep4_dev = [e for e in ep4 if dev_start <= e.start <= dev_end]

alert_set = set(det_df[det_df["vix"] == 1]["week"])
persist_set = set(det_df[det_df["persist"] == 1]["week"])
always_set = set(det_df["week"])

ep_res_vix2 = detect_event_results(ep2_dev, alert_set)
ep_res_pers2 = detect_event_results(ep2_dev, persist_set)
ep_res_af2 = detect_event_results(ep2_dev, always_set)
ep_res_vix0 = detect_event_results(ep0_dev, alert_set)
ep_res_vix4 = detect_event_results(ep4_dev, alert_set)

hit_vix2 = sum(r.hit for r in ep_res_vix2)
hit_pers2 = sum(r.hit for r in ep_res_pers2)
hit_af2 = sum(r.hit for r in ep_res_af2)
hit_vix0 = sum(r.hit for r in ep_res_vix0)
hit_vix4 = sum(r.hit for r in ep_res_vix4)

print(f"VIX event recall (mg=2): {hit_vix2}/{len(ep2_dev)}")

# ── Predict event metrics (VIX rule, calm-at-t) ───────────────────────────────
calm_set_vix = set(det_df[(det_df["vix"] == 1) & (det_df["detect"] == 0)]["week"])
calm_set_pers = set(det_df[(det_df["persist"] == 1) & (det_df["detect"] == 0)]["week"])
calm_set_always = set(det_df[det_df["detect"] == 0]["week"])

pred_res_vix = predict_event_results(ep2_dev, calm_set_vix, horizon=4)
pred_res_pers = predict_event_results(ep2_dev, calm_set_pers, horizon=4)
pred_res_af = predict_event_results(ep2_dev, calm_set_always, horizon=4)

# ── False alarm runs ──────────────────────────────────────────────────────────
dev_weeks = [w for w in weeks if dev_start <= w <= dev_end]
fa_vix = false_alarm_runs(
    sorted(alert_set & set(dev_weeks)),
    dev_weeks,
    ep2_dev,
    grace_weeks=2,
    known_unlabeled=KNOWN_UNLABELED,
)
fa_pers = false_alarm_runs(
    sorted(persist_set & set(dev_weeks)),
    dev_weeks,
    ep2_dev,
    grace_weeks=2,
    known_unlabeled=KNOWN_UNLABELED,
)

fa_vix_real = [r for r in fa_vix if not r.in_known_unlabeled]
fa_pers_real = [r for r in fa_pers if not r.in_known_unlabeled]

ep2_dev_years: set[int] = set()
for ep in ep2_dev:
    d2 = ep.start
    while d2 <= ep.end:
        ep2_dev_years.add(d2.year)
        d2 += timedelta(weeks=1)
calm_dev_years = [y for y in range(2008, 2018) if y not in ep2_dev_years]
fa_vix_calm = [r for r in fa_vix_real if r.year in calm_dev_years]
fa_pers_calm = [r for r in fa_pers_real if r.year in calm_dev_years]


# ── Write report ──────────────────────────────────────────────────────────────
def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n // d}%)" if d else "0/0"


today_str = date.today().isoformat()
out: list[str] = []

out.append("# Baselines Report — ews_v1 Development Folds (2008–2017)")
out.append("")
out.append(f"Generated: {today_str}  ")
out.append("Data source: `data/raw/ews_weekly_2000_2021.csv` (EWS Bloomberg labels)  ")
out.append("Label set: `ews_v1` · Horizon h = 4 · Merge gap = 2 · FAR budget = 5%  ")
out.append("Development folds: test years 2008–2017 (expanding window, val ≥ 8 positives)  ")
out.append("")
out.append("> **Note:** The 2018–2021 confirmation window contains only 6 episodes")
out.append("> (2018-10, 2018-12, 2020-02, 2020-06, 2020-08, 2020-10) and will be run")
out.append("> once at the end of Phase 2 Step 5. These results are development-only.")
out.append("")
out.append("---")
out.append("")
out.append("## Summary")
out.append("")
tot_w = len(det_df)
anom_w = int(det_df["detect"].sum())
out.append(f"- Out-of-sample dev weeks: {tot_w}  ")
out.append(f"- Anomaly weeks: {anom_w} ({100 * anom_w / tot_w:.1f}%)  ")
out.append(f"- Episodes (merge\\_gap=2, dev): {len(ep2_dev)}  ")
out.append(
    f"- Calm-at-t predict positives: {calm_at_t_pos}"
    f" (of {len(calm_pred_df)} calm weeks with predict label)  "
)
out.append("")
out.append("---")
out.append("")
out.append("## Row-level metrics — detect head")
out.append("")
out.append("| Baseline | Prec | Recall | F1 | FAR | TP | FP | FN |")
out.append("|----------|------|--------|-----|-----|----|----|-----|")
for b in baselines:
    m = detect_metrics[b]
    out.append(
        f"| {baseline_names[b]} | {m['precision']:.3f} | {m['recall']:.3f}"
        f" | {m['f1']:.3f} | {m['false_alarm_rate']:.3f}"
        f" | {m['tp']} | {m['fp']} | {m['fn']} |"
    )
out.append("")
out.append("---")
out.append("")
out.append("## Event-level metrics — detect head")
out.append("")
out.append("### Per-episode hit/miss/lead-time (merge\\_gap = 2, VIX rule)")
out.append("")
out.append("| # | Start | End | Weeks | VIX hit | VIX weeks-in | Persist hit | Always hit |")
out.append("|---|-------|-----|-------|---------|--------------|-------------|------------|")
for i, (rv, rp, ra) in enumerate(zip(ep_res_vix2, ep_res_pers2, ep_res_af2), 1):
    ep = rv.episode
    vix_det = "Y" if rv.hit else "N"
    vix_wi = str(rv.weeks_into_episode) if rv.hit else "-"
    pers_det = "Y" if rp.hit else "N"
    af_det = "Y" if ra.hit else "N"
    out.append(
        f"| {i} | {ep.start} | {ep.end} | {ep.length}"
        f" | {vix_det} | {vix_wi} | {pers_det} | {af_det} |"
    )
out.append("")
out.append(
    f"**Event recall (merge\\_gap=2):** VIX {pct(hit_vix2, len(ep2_dev))},"
    f" Persistence {pct(hit_pers2, len(ep2_dev))},"
    f" Always-flag {pct(hit_af2, len(ep2_dev))}"
)
out.append("")
out.append("### Sensitivity to merge gap (VIX rule, detect head)")
out.append("")
out.append("| Merge gap | Episodes | VIX hits | Recall |")
out.append("|-----------|----------|----------|--------|")
for mg, eps, hits in [(0, ep0_dev, hit_vix0), (2, ep2_dev, hit_vix2), (4, ep4_dev, hit_vix4)]:
    out.append(f"| {mg} | {len(eps)} | {hits} | {hits / len(eps):.3f} |")
out.append("")
out.append("---")
out.append("")
out.append("## Predict head — calm-at-t analysis")
out.append("")
out.append(
    f"Evaluated on {len(calm_pred_df)} calm-at-t weeks (detect==0 with valid predict label).  "
)
out.append(f"Calm-at-t positives (predict==1 while calm): **{calm_at_t_pos}**")
out.append("")
out.append("### Row-level metrics (calm-at-t weeks only)")
out.append("")
out.append("| Baseline | Prec | Recall | F1 | FAR |")
out.append("|----------|------|--------|-----|-----|")
for b in baselines:
    m = predict_metrics[b]
    out.append(
        f"| {baseline_names[b]} | {m['precision']:.3f} | {m['recall']:.3f}"
        f" | {m['f1']:.3f} | {m['false_alarm_rate']:.3f} |"
    )
out.append("")
out.append("### Event-level advance warning (VIX rule, calm-at-t alerts, h=4)")
out.append("")
hit_pv = sum(r.hit for r in pred_res_vix)
hit_pp = sum(r.hit for r in pred_res_pers)
hit_paf = sum(r.hit for r in pred_res_af)
out.append(f"VIX rule: {pct(hit_pv, len(ep2_dev))} episodes warned  ")
out.append(f"Persistence: {pct(hit_pp, len(ep2_dev))} episodes warned  ")
out.append(f"Always-flag (calm): {pct(hit_paf, len(ep2_dev))} episodes warned")
out.append("")
out.append("| # | Episode start | VIX warned | VIX lead (w) | Persist warned | Always warned |")
out.append("|---|--------------|------------|--------------|----------------|---------------|")
for i, (rv, rp, ra) in enumerate(zip(pred_res_vix, pred_res_pers, pred_res_af), 1):
    ep = rv.episode
    vw = "Y" if rv.hit else "N"
    vl = str(rv.weeks_into_episode) if rv.hit else "-"
    pw = "Y" if rp.hit else "N"
    aw = "Y" if ra.hit else "N"
    out.append(f"| {i} | {ep.start} | {vw} | {vl} | {pw} | {aw} |")
out.append("")
out.append("---")
out.append("")
out.append("## False alarm runs (detect head, calm-year metric)")
out.append("")
out.append(
    "Grace window: ±2 weeks around each episode. Known-unlabeled windows excluded from count."
)
out.append(f"Calm dev years (no labeled episode): {calm_dev_years}")
out.append("")
out.append("| Baseline | Total FA runs | FA in calm years | FA runs in known-unlabeled |")
out.append("|----------|--------------|-----------------|---------------------------|")
ku_vix_count = sum(1 for r in fa_vix if r.in_known_unlabeled)
ku_pers_count = sum(1 for r in fa_pers if r.in_known_unlabeled)
out.append(f"| VIX rule | {len(fa_vix_real)} | {len(fa_vix_calm)} | {ku_vix_count} |")
out.append(f"| Persistence | {len(fa_pers_real)} | {len(fa_pers_calm)} | {ku_pers_count} |")
out.append("| Always-flag | — | — | — (all weeks flagged) |")
out.append("")
out.append("---")
out.append("")
out.append("## Known events with no EWS label")
out.append("")
out.append("Alerts in these windows are reported below but excluded from false-alarm counts.")
out.append("")
out.append("| Event | Window | VIX alerts | Persist alerts |")
out.append("|-------|--------|------------|----------------|")
weeks_set = set(weeks)
for ks, ke, kname in KNOWN_UNLABELED:
    vix_c = sum(1 for w in weeks if ks <= w <= ke and w in alert_set)
    pers_c = sum(1 for w in weeks if ks <= w <= ke and w in persist_set)
    out.append(f"| {kname} | {ks} – {ke} | {vix_c} | {pers_c} |")
out.append("")
out.append("---")
out.append("")
out.append("## Baselines to beat")
out.append("")
out.append("A candidate model must exceed these bars on the development folds:")
out.append("")
out.append(
    f"- **Event recall (detect, mg=2):** VIX rule ="
    f" {hit_vix2}/{len(ep2_dev)} ({100 * hit_vix2 // len(ep2_dev)}%)"
)
out.append(f"- **Row-level F1 (detect):** VIX rule = {detect_metrics['vix']['f1']:.3f}")
out.append(
    f"- **Calm-year false alarm runs:** VIX rule = {len(fa_vix_calm)}"
    f" in {len(calm_dev_years)} calm years"
)
out.append(
    f"- **Predict advance-warning recall:** VIX rule ="
    f" {hit_pv}/{len(ep2_dev)} ({100 * hit_pv // len(ep2_dev)}%)"
)
out.append("")
out.append(
    "The VIX rule is the bar. A model that does not beat the VIX rule on event recall "
    "is not an improvement over a simple threshold rule."
)
out.append("")

report_path = f"docs/reports/{today_str}-baselines.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print(f"Written: {report_path}")
