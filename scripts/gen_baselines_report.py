"""Generate the baselines report for docs/reports/<date>-baselines.md.

Uses VIX level and VIX z52 (52-week rolling z-score of log VIX level) from
the EWS CSV as reference numbers.  The production `mad-model baselines` command
reads VIX from the feature pipeline DB; this script provides a CSV-based sanity
check against the expected figures:
  vix_level: F1 ≈ 0.56, 5/14 episodes
  vix_z52:   F1 ≈ 0.27, 4/14 episodes
"""

from __future__ import annotations

import math
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from market_anomaly_detector.episodes import extract_episodes
from market_anomaly_detector.eval_harness import (
    KNOWN_UNLABELED,
    choose_threshold,
    detect_event_results,
    false_alarm_runs,
    make_folds_v2,
    persistence,
    predict_event_results,
    row_metrics,
    vix_rule,
    vix_z52,
)
from market_anomaly_detector.labels import build_targets

# ── Load EWS data ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/raw/ews_weekly_2000_2021.csv", parse_dates=["Data"])
df = df.sort_values("Data").reset_index(drop=True)
weeks: list[date] = [d.date() for d in df["Data"]]
detect = df["Y"].values.astype(int)

vix_close = df["VIX"].values.astype(float)
vix_level_scores = vix_close
vix_z52_scores = vix_z52(vix_close)

labels_df = pd.DataFrame({"week": weeks, "value": detect})
targets = build_targets(labels_df, horizon=4)
predict_labels = targets["predict"].values

ep2 = extract_episodes(labels_df, merge_gap=2)
ep0 = extract_episodes(labels_df, merge_gap=0)
ep4 = extract_episodes(labels_df, merge_gap=4)

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
    val_ku = ku_mask[fold.val_lo : fold.test_lo]

    thr_level = choose_threshold(
        val_detect,
        vix_level_scores[fold.val_lo : fold.test_lo],
        far_budget=0.05,
        excluded_mask=val_ku,
    )
    thr_z52 = choose_threshold(
        val_detect,
        vix_z52_scores[fold.val_lo : fold.test_lo],
        far_budget=0.05,
        excluded_mask=val_ku,
    )

    test_slice = slice(fold.test_lo, fold.test_hi)
    t_detect = detect[test_slice]
    t_vix_level = vix_level_scores[test_slice]
    t_vix_z52 = vix_z52_scores[test_slice]
    t_pred_labels = predict_labels[test_slice]
    t_weeks = weeks[fold.test_lo : fold.test_hi]

    per = persistence(t_detect)
    vrl = vix_rule(t_vix_level, thr_level)
    vrz = vix_rule(t_vix_z52, thr_z52)

    for k, (w, dl, pl) in enumerate(zip(t_weeks, t_detect, t_pred_labels)):
        rows_detect.append(
            {
                "week": w,
                "detect": int(dl),
                "always": 1,
                "persist": int(per[k]),
                "vix_level": int(vrl[k]),
                "vix_z52": int(vrz[k]),
                "vix_close": float(t_vix_level[k]),
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
                    "vix_level": int(vrl[k]),
                    "vix_z52": int(vrz[k]),
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
all_baselines = ["always", "persist", "vix_level", "vix_z52"]
bnames = {
    "always": "Always-flag",
    "persist": "Persistence (oracle ref.)",
    "vix_level": "VIX level",
    "vix_z52": "VIX z52 (log level)",
}
detect_metrics = {b: row_metrics(det_df["detect"], det_df[b]) for b in all_baselines}
predict_metrics = {b: row_metrics(calm_pred_df["predict"], calm_pred_df[b]) for b in all_baselines}

# ── Event-level metrics ───────────────────────────────────────────────────────
dev_start, dev_end = date(2008, 1, 1), date(2017, 12, 31)
ep2_dev = [e for e in ep2 if dev_start <= e.start <= dev_end]
ep0_dev = [e for e in ep0 if dev_start <= e.start <= dev_end]
ep4_dev = [e for e in ep4 if dev_start <= e.start <= dev_end]

alert_level = set(det_df[det_df["vix_level"] == 1]["week"])
alert_z52 = set(det_df[det_df["vix_z52"] == 1]["week"])
persist_set = set(det_df[det_df["persist"] == 1]["week"])
always_set = set(det_df["week"])

ep_res_level2 = detect_event_results(ep2_dev, alert_level)
ep_res_z52_2 = detect_event_results(ep2_dev, alert_z52)
ep_res_pers2 = detect_event_results(ep2_dev, persist_set)
ep_res_af2 = detect_event_results(ep2_dev, always_set)
ep_res_level0 = detect_event_results(ep0_dev, alert_level)
ep_res_level4 = detect_event_results(ep4_dev, alert_level)
ep_res_z52_0 = detect_event_results(ep0_dev, alert_z52)
ep_res_z52_4 = detect_event_results(ep4_dev, alert_z52)

hit_level2 = sum(r.hit for r in ep_res_level2)
hit_z52_2 = sum(r.hit for r in ep_res_z52_2)
hit_pers2 = sum(r.hit for r in ep_res_pers2)
hit_af2 = sum(r.hit for r in ep_res_af2)

print(f"VIX level event recall (mg=2): {hit_level2}/{len(ep2_dev)}")
print(f"VIX z52 event recall (mg=2):   {hit_z52_2}/{len(ep2_dev)}")

# ── Predict event metrics ──────────────────────────────────────────────────────
calm_level = set(det_df[(det_df["vix_level"] == 1) & (det_df["detect"] == 0)]["week"])
calm_z52 = set(det_df[(det_df["vix_z52"] == 1) & (det_df["detect"] == 0)]["week"])
calm_pers = set(det_df[(det_df["persist"] == 1) & (det_df["detect"] == 0)]["week"])
calm_always = set(det_df[det_df["detect"] == 0]["week"])

pred_res_level = predict_event_results(ep2_dev, calm_level, horizon=4)
pred_res_z52 = predict_event_results(ep2_dev, calm_z52, horizon=4)
pred_res_pers = predict_event_results(ep2_dev, calm_pers, horizon=4)
pred_res_af = predict_event_results(ep2_dev, calm_always, horizon=4)

# ── False alarm runs ──────────────────────────────────────────────────────────
dev_weeks = [w for w in weeks if dev_start <= w <= dev_end]
fa_level = false_alarm_runs(
    sorted(alert_level & set(dev_weeks)),
    dev_weeks,
    ep2_dev,
    grace_weeks=2,
    known_unlabeled=KNOWN_UNLABELED,
)
fa_z52 = false_alarm_runs(
    sorted(alert_z52 & set(dev_weeks)),
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

fa_level_real = [r for r in fa_level if not r.in_known_unlabeled]
fa_z52_real = [r for r in fa_z52 if not r.in_known_unlabeled]
fa_pers_real = [r for r in fa_pers if not r.in_known_unlabeled]

ep2_dev_years: set[int] = set()
for ep in ep2_dev:
    d2 = ep.start
    while d2 <= ep.end:
        ep2_dev_years.add(d2.year)
        d2 += timedelta(weeks=1)
calm_dev_years = [y for y in range(2008, 2018) if y not in ep2_dev_years]
fa_level_calm = [r for r in fa_level_real if r.year in calm_dev_years]
fa_z52_calm = [r for r in fa_z52_real if r.year in calm_dev_years]

# Episode VIX stats
vix_by_week = {w: v for w, v in zip(weeks, vix_close)}
ep_vix_stats: list[dict] = []
for ep in ep2_dev:
    closes: list[float] = []
    d3 = ep.start
    while d3 <= ep.end:
        v = vix_by_week.get(d3, float("nan"))
        if not math.isnan(v):
            closes.append(v)
        d3 += timedelta(weeks=1)
    ep_vix_stats.append(
        {
            "ep": ep,
            "median": float(np.median(closes)) if closes else float("nan"),
            "max": float(np.max(closes)) if closes else float("nan"),
        }
    )


# ── Write report ──────────────────────────────────────────────────────────────
def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n // d}%)" if d else "0/0"


today_str = date.today().isoformat()
out: list[str] = []

out.append("# Baselines Report — ews_v1 Development Folds (2008–2017)")
out.append("")
out.append(f"Generated: {today_str}  ")
out.append("VIX source: `data/raw/ews_weekly_2000_2021.csv` (CSV reference — prod uses DB)  ")
out.append("Label set: `ews_v1` · Horizon h = 4 · Merge gap = 2 · FAR budget = 5%  ")
out.append("Development folds: test years 2008–2017 (expanding window, val ≥ 8 positives)  ")
out.append("")
out.append("> **Note:** The 2018–2021 confirmation window contains only 6 episodes")
out.append("> (2018-10, 2018-12, 2020-02, 2020-06, 2020-08, 2020-10) and will be run")
out.append("> once at the end of Phase 2 Step 5. These results are development-only.")
out.append("")
out.append("> **Persistence** is an oracle reference baseline: it requires last week's")
out.append("> label, which is not available at live scoring time. It is reported for")
out.append("> context but excluded from the model target bar.")
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
for b in all_baselines:
    m = detect_metrics[b]
    suffix = " *(oracle)*" if b == "persist" else ""
    out.append(
        f"| {bnames[b]}{suffix} | {m['precision']:.3f} | {m['recall']:.3f}"
        f" | {m['f1']:.3f} | {m['false_alarm_rate']:.3f}"
        f" | {m['tp']} | {m['fp']} | {m['fn']} |"
    )
out.append("")
out.append("---")
out.append("")
out.append("## Event-level metrics — detect head")
out.append("")
out.append("### Per-episode hit/miss/lead-time (merge\\_gap = 2)")
out.append("")
out.append(
    "| # | Start | End | Wks | Level hit | Level w-in | z52 hit | z52 w-in | Persist† | Always |"
)
out.append(
    "|---|-------|-----|-----|-----------|-----------|---------|---------|---------|--------|"
)
for i, (rl, rz, rp, ra) in enumerate(zip(ep_res_level2, ep_res_z52_2, ep_res_pers2, ep_res_af2), 1):
    ep = rl.episode
    out.append(
        f"| {i} | {ep.start} | {ep.end} | {ep.length}"
        f" | {'Y' if rl.hit else 'N'} | {rl.weeks_into_episode if rl.hit else '-'}"
        f" | {'Y' if rz.hit else 'N'} | {rz.weeks_into_episode if rz.hit else '-'}"
        f" | {'Y' if rp.hit else 'N'} | {'Y' if ra.hit else 'N'} |"
    )
out.append("")
out.append("† Persistence is an oracle reference (last week's label).")
out.append("")
out.append(
    f"**Event recall (merge\\_gap=2):**"
    f" VIX level {pct(hit_level2, len(ep2_dev))},"
    f" VIX z52 {pct(hit_z52_2, len(ep2_dev))},"
    f" Persistence† {pct(hit_pers2, len(ep2_dev))},"
    f" Always-flag {pct(hit_af2, len(ep2_dev))}"
)
out.append("")
out.append("### Sensitivity to merge gap")
out.append("")
out.append("| Merge gap | Episodes | Level hits | Level recall | z52 hits | z52 recall |")
out.append("|-----------|----------|-----------|--------------|---------|-----------|")
for mg, eps, rls, rzs in [
    (0, ep0_dev, ep_res_level0, ep_res_z52_0),
    (2, ep2_dev, ep_res_level2, ep_res_z52_2),
    (4, ep4_dev, ep_res_level4, ep_res_z52_4),
]:
    hl = sum(r.hit for r in rls)
    hz = sum(r.hit for r in rzs)
    out.append(f"| {mg} | {len(eps)} | {hl} | {hl / len(eps):.3f} | {hz} | {hz / len(eps):.3f} |")
out.append("")
out.append("---")
out.append("")
out.append("## VIX level inside dev episodes")
out.append("")
out.append(
    "Episodes with median VIX < 20 cannot be caught by any VIX threshold rule (marked **low-VIX**)."
)
out.append("")
out.append("| # | Start | End | Wks | VIX median | VIX max | Catchable |")
out.append("|---|-------|-----|-----|-----------|---------|-----------|")
for i, s in enumerate(ep_vix_stats, 1):
    ep = s["ep"]
    med = s["median"]
    mx = s["max"]
    catchable = "N — low-VIX" if (not math.isnan(med) and med < 20) else "Y"
    med_str = f"{med:.1f}" if not math.isnan(med) else "—"
    mx_str = f"{mx:.1f}" if not math.isnan(mx) else "—"
    out.append(
        f"| {i} | {ep.start} | {ep.end} | {ep.length} | {med_str} | {mx_str} | {catchable} |"
    )
low_vix_count = sum(1 for s in ep_vix_stats if not math.isnan(s["median"]) and s["median"] < 20)
out.append(
    f"\n{low_vix_count} of {len(ep2_dev)} dev episodes have median VIX < 20;"
    " no level threshold can catch them."
)
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
for b in all_baselines:
    m = predict_metrics[b]
    suffix = " *(oracle)*" if b == "persist" else ""
    out.append(
        f"| {bnames[b]}{suffix} | {m['precision']:.3f} | {m['recall']:.3f}"
        f" | {m['f1']:.3f} | {m['false_alarm_rate']:.3f} |"
    )
out.append("")
out.append("### Event-level advance warning (calm-at-t alerts, h=4)")
out.append("")
hit_pl = sum(r.hit for r in pred_res_level)
hit_pz = sum(r.hit for r in pred_res_z52)
hit_pp = sum(r.hit for r in pred_res_pers)
hit_paf = sum(r.hit for r in pred_res_af)
out.append(f"VIX level: {pct(hit_pl, len(ep2_dev))} episodes warned  ")
out.append(f"VIX z52: {pct(hit_pz, len(ep2_dev))} episodes warned  ")
out.append(f"Persistence†: {pct(hit_pp, len(ep2_dev))} episodes warned  ")
out.append(f"Always-flag (calm): {pct(hit_paf, len(ep2_dev))} episodes warned")
out.append("")
out.append(
    "| # | Ep start | Level warned | Level lead (w)"
    " | z52 warned | z52 lead (w) | Persist† | Always |"
)
out.append(
    "|---|----------|-------------|----------------|-----------|-------------|---------|--------|"
)
for i, (rl, rz, rp, ra) in enumerate(
    zip(pred_res_level, pred_res_z52, pred_res_pers, pred_res_af), 1
):
    ep = rl.episode
    out.append(
        f"| {i} | {ep.start}"
        f" | {'Y' if rl.hit else 'N'} | {rl.weeks_into_episode if rl.hit else '-'}"
        f" | {'Y' if rz.hit else 'N'} | {rz.weeks_into_episode if rz.hit else '-'}"
        f" | {'Y' if rp.hit else 'N'} | {'Y' if ra.hit else 'N'} |"
    )
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
out.append("| Baseline | Total FA runs | FA in calm years | FA in known-unlabeled |")
out.append("|----------|--------------|-----------------|----------------------|")
ku_level = sum(1 for r in fa_level if r.in_known_unlabeled)
ku_z52 = sum(1 for r in fa_z52 if r.in_known_unlabeled)
ku_pers = sum(1 for r in fa_pers if r.in_known_unlabeled)
out.append(f"| VIX level | {len(fa_level_real)} | {len(fa_level_calm)} | {ku_level} |")
out.append(f"| VIX z52 | {len(fa_z52_real)} | {len(fa_z52_calm)} | {ku_z52} |")
out.append(f"| Persistence† | {len(fa_pers_real)} | — | {ku_pers} |")
out.append("| Always-flag | — | — | — (all weeks flagged) |")
out.append("")
out.append("---")
out.append("")
out.append("## Known events with no EWS label")
out.append("")
out.append("Alerts in these windows are reported below but excluded from false-alarm counts.")
out.append("")
out.append("| Event | Window | Level alerts | z52 alerts |")
out.append("|-------|--------|-------------|------------|")
for ks, ke, kname in KNOWN_UNLABELED:
    lc = sum(1 for w in weeks if ks <= w <= ke and w in alert_level)
    zc = sum(1 for w in weeks if ks <= w <= ke and w in alert_z52)
    out.append(f"| {kname} | {ks} – {ke} | {lc} | {zc} |")
out.append("")
out.append("---")
out.append("")
out.append("## Baselines to beat")
out.append("")
out.append(
    "A candidate model must exceed **both** VIX level and VIX z52 on the development"
    " folds. Persistence is excluded (oracle reference — not achievable live)."
)
out.append("")
best_recall = max(hit_level2, hit_z52_2)
best_f1 = max(detect_metrics["vix_level"]["f1"], detect_metrics["vix_z52"]["f1"])
best_pred = max(hit_pl, hit_pz)
out.append(
    f"- **Event recall (detect, mg=2):**"
    f" VIX level {pct(hit_level2, len(ep2_dev))},"
    f" VIX z52 {pct(hit_z52_2, len(ep2_dev))}"
    f" → beat {pct(best_recall, len(ep2_dev))}"
)
out.append(
    f"- **Row-level F1 (detect):**"
    f" VIX level {detect_metrics['vix_level']['f1']:.3f},"
    f" VIX z52 {detect_metrics['vix_z52']['f1']:.3f}"
    f" → beat {best_f1:.3f}"
)
out.append(
    f"- **Calm-year false alarm runs:**"
    f" VIX level {len(fa_level_calm)}, VIX z52 {len(fa_z52_calm)}"
    f" in {len(calm_dev_years)} calm years"
)
out.append(
    f"- **Predict advance-warning recall:**"
    f" VIX level {pct(hit_pl, len(ep2_dev))},"
    f" VIX z52 {pct(hit_pz, len(ep2_dev))}"
    f" → beat {pct(best_pred, len(ep2_dev))}"
)
out.append("")

report_path = f"docs/reports/{today_str}-baselines.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print(f"Written: {report_path}")
