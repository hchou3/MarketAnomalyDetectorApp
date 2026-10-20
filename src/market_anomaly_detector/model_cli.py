"""mad-model CLI — model management commands."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

_REPORTS_DIR = Path(__file__).parent.parent.parent / "docs" / "reports"

_DB_NOT_INIT_MSG = "Database not initialized — run: uv run alembic upgrade head"

# Known historical crisis windows used to annotate episodes in the report.
# Each entry: (start_iso, end_iso, label).
_KNOWN_CRISES: list[tuple[str, str, str]] = [
    ("2007-06-01", "2007-12-31", "2007 subprime onset"),
    ("2008-01-01", "2009-06-30", "2008 financial crisis"),
    ("2010-04-01", "2010-09-30", "2010 Eurozone / flash crash"),
    ("2011-06-01", "2012-06-30", "2011–12 Euro debt crisis"),
    ("2013-05-01", "2013-08-31", "2013 taper tantrum"),
    ("2014-09-01", "2015-02-28", "2014–15 oil shock"),
    ("2015-06-01", "2016-04-30", "2015–16 China slowdown / oil crash"),
    ("2018-10-01", "2019-02-28", "2018 Q4 global equity selloff"),
    ("2020-01-01", "2020-12-31", "2020 COVID-19"),
]


def _historical_match(start: date, end: date) -> str:
    """Return the best-matching known crisis label for an episode, or empty string."""
    best_label = ""
    best_overlap = 0
    ep_start = start.toordinal()
    ep_end = end.toordinal()
    for cs, ce, label in _KNOWN_CRISES:
        cs_ord = date.fromisoformat(cs).toordinal()
        ce_ord = date.fromisoformat(ce).toordinal()
        overlap = max(0, min(ep_end, ce_ord) - max(ep_start, cs_ord))
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = label
    return best_label


def _require_db_ready(engine) -> None:
    from sqlalchemy import inspect as sa_inspect

    insp = sa_inspect(engine)
    if not insp.has_table("instruments"):
        print(_DB_NOT_INIT_MSG, file=sys.stderr)
        raise SystemExit(2)


# ── episodes ──────────────────────────────────────────────────────────────────


def _cmd_episodes(args: argparse.Namespace) -> int:
    from sqlalchemy.orm import Session

    from .db.engine import create_engine
    from .episodes import extract_episodes
    from .labels import load_labels

    label_set: str = args.label_set
    merge_gap: int = args.merge_gap

    try:
        engine = create_engine(url=args.database_url)
        with Session(engine) as session:
            _require_db_ready(engine)
            labels = load_labels(session, label_set=label_set)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"ERROR: database error: {exc}", file=sys.stderr)
        return 1

    if labels.empty:
        print(f"ERROR: no labels found for label_set={label_set!r}", file=sys.stderr)
        return 1

    episodes_by_gap = {g: extract_episodes(labels, merge_gap=g) for g in (0, 2, 4)}
    main_episodes = extract_episodes(labels, merge_gap=merge_gap)

    out_path = _REPORTS_DIR / f"episodes-{label_set}.md"
    _write_report(out_path, label_set, merge_gap, main_episodes, labels, episodes_by_gap)

    print(
        f"episodes: {len(main_episodes)} episodes (merge_gap={merge_gap})"
        f" → {out_path.relative_to(Path(__file__).parent.parent.parent)}"
    )
    return 0


def _write_report(
    out_path: Path,
    label_set: str,
    merge_gap: int,
    episodes,
    labels,
    episodes_by_gap: dict[int, list],
) -> None:
    today = date.today().isoformat()
    label_count = int((labels["value"] == 1).sum())
    total_weeks = len(labels)
    date_range = f"{labels['week'].iloc[0]} – {labels['week'].iloc[-1]}"

    lines: list[str] = [
        f"# Anomaly Episodes — {label_set}",
        "",
        f"Generated: {today}  ",
        f"Label set: `{label_set}`  ",
        f"Date range: {date_range}  ",
        f"Total weeks: {total_weeks}  ",
        f"Anomaly weeks: {label_count} ({100 * label_count / total_weeks:.1f}%)",
        "",
        f"## Episode list (merge gap = {merge_gap} weeks)",
        "",
        "| # | Start | End | Weeks | Historical match |",
        "|---|-------|-----|-------|-----------------|",
    ]

    for i, ep in enumerate(episodes, 1):
        match = _historical_match(ep.start, ep.end)
        lines.append(f"| {i} | {ep.start} | {ep.end} | {ep.length} | {match} |")

    lines += [
        "",
        f"**Total: {len(episodes)} episode(s)**",
        "",
        "## Episode count by merge gap",
        "",
        "| Merge gap (calm weeks) | Episodes |",
        "|------------------------|----------|",
    ]
    for g in (0, 2, 4):
        lines.append(f"| {g} | {len(episodes_by_gap[g])} |")

    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ── baselines ─────────────────────────────────────────────────────────────────


def _cmd_baselines(args: argparse.Namespace) -> int:
    import math
    from datetime import timedelta

    import numpy as np
    import pandas as pd
    from sqlalchemy.orm import Session

    from .db.engine import create_engine
    from .episodes import extract_episodes
    from .eval_harness import (
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
    from .features.build import weekly_close
    from .labels import build_targets, load_labels

    # ── Load from DB ──────────────────────────────────────────────────────────
    try:
        engine = create_engine(url=args.database_url)
        with Session(engine) as session:
            _require_db_ready(engine)
            labels_df = load_labels(session, label_set=args.label_set)
            if labels_df.empty:
                print(f"ERROR: no labels for {args.label_set!r}", file=sys.stderr)
                return 1
            tuesdays = list(labels_df["week"])
            vix_close_arr = weekly_close(session, "vix", tuesdays)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    weeks = tuesdays
    detect = labels_df["value"].values.astype(int)
    vix_level_scores = vix_close_arr
    vix_z52_scores = vix_z52(vix_close_arr)

    labels_for_targets = pd.DataFrame({"week": weeks, "value": detect})
    targets = build_targets(labels_for_targets, horizon=args.horizon)
    predict_labels = targets["predict"].values

    ep2 = extract_episodes(labels_for_targets, merge_gap=2)
    ep0 = extract_episodes(labels_for_targets, merge_gap=0)
    ep4 = extract_episodes(labels_for_targets, merge_gap=4)

    ku_mask = np.zeros(len(weeks), dtype=bool)
    for ks, ke, _ in KNOWN_UNLABELED:
        for i, w in enumerate(weeks):
            if ks <= w <= ke:
                ku_mask[i] = True

    folds = make_folds_v2(
        weeks,
        detect,
        first_test_year=args.first_test_year,
        last_test_year=args.last_test_year,
    )

    # ── Walk-forward evaluation ───────────────────────────────────────────────
    rows_detect: list[dict] = []
    rows_predict: list[dict] = []
    for fold in folds:
        val_detect = detect[fold.val_lo : fold.test_lo]
        val_ku = ku_mask[fold.val_lo : fold.test_lo]

        thr_level = choose_threshold(
            val_detect,
            vix_level_scores[fold.val_lo : fold.test_lo],
            far_budget=args.far_budget,
            excluded_mask=val_ku,
        )
        thr_z52 = choose_threshold(
            val_detect,
            vix_z52_scores[fold.val_lo : fold.test_lo],
            far_budget=args.far_budget,
            excluded_mask=val_ku,
        )

        t_detect = detect[fold.test_lo : fold.test_hi]
        t_vix_level = vix_level_scores[fold.test_lo : fold.test_hi]
        t_vix_z52 = vix_z52_scores[fold.test_lo : fold.test_hi]
        t_pred = predict_labels[fold.test_lo : fold.test_hi]
        t_weeks = weeks[fold.test_lo : fold.test_hi]

        per = persistence(t_detect)
        vrl = vix_rule(t_vix_level, thr_level)
        vrz = vix_rule(t_vix_z52, thr_z52)

        for k, (wk, dl, pl) in enumerate(zip(t_weeks, t_detect, t_pred)):
            rows_detect.append(
                {
                    "week": wk,
                    "detect": int(dl),
                    "always": 1,
                    "persist": int(per[k]),
                    "vix_level": int(vrl[k]),
                    "vix_z52": int(vrz[k]),
                    "vix_close": float(t_vix_level[k]),
                    "fold": fold.test_year,
                }
            )
            if not math.isnan(pl):
                rows_predict.append(
                    {
                        "week": wk,
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

    all_baselines = ["always", "persist", "vix_level", "vix_z52"]
    bnames = {
        "always": "Always-flag",
        "persist": "Persistence (oracle ref.)",
        "vix_level": "VIX level",
        "vix_z52": "VIX z52 (log level)",
    }
    detect_metrics = {b: row_metrics(det_df["detect"], det_df[b]) for b in all_baselines}
    predict_metrics = {
        b: row_metrics(calm_pred_df["predict"], calm_pred_df[b]) for b in all_baselines
    }

    first_test = args.first_test_year
    last_test = args.last_test_year or max(w.year for w in weeks)
    dev_start = date(first_test, 1, 1)
    dev_end = date(last_test, 12, 31)

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

    calm_level = set(det_df[(det_df["vix_level"] == 1) & (det_df["detect"] == 0)]["week"])
    calm_z52 = set(det_df[(det_df["vix_z52"] == 1) & (det_df["detect"] == 0)]["week"])
    calm_pers = set(det_df[(det_df["persist"] == 1) & (det_df["detect"] == 0)]["week"])
    calm_always = set(det_df[det_df["detect"] == 0]["week"])

    pred_res_level = predict_event_results(ep2_dev, calm_level, horizon=args.horizon)
    pred_res_z52 = predict_event_results(ep2_dev, calm_z52, horizon=args.horizon)
    pred_res_pers = predict_event_results(ep2_dev, calm_pers, horizon=args.horizon)
    pred_res_af = predict_event_results(ep2_dev, calm_always, horizon=args.horizon)

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
    calm_dev_years = [y for y in range(first_test, last_test + 1) if y not in ep2_dev_years]
    fa_level_calm = [r for r in fa_level_real if r.year in calm_dev_years]
    fa_z52_calm = [r for r in fa_z52_real if r.year in calm_dev_years]

    # Episode VIX stats for the "catchable" analysis
    vix_by_week = {w: v for w, v in zip(weeks, vix_level_scores)}
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

    def pct(n: int, d: int) -> str:
        return f"{n}/{d} ({100 * n // d}%)" if d else "0/0"

    # ── Write report ──────────────────────────────────────────────────────────
    today_str = date.today().isoformat()
    out: list[str] = []
    out.append(
        f"# Baselines Report — {args.label_set} Development Folds ({first_test}–{last_test})"
    )
    out.append("")
    out.append(f"Generated: {today_str}  ")
    out.append("VIX source: feature pipeline / DB (Yahoo `^VIX`, known\\_at-correct)  ")
    out.append(
        f"Label set: `{args.label_set}` · Horizon h = {args.horizon}"
        f" · Merge gap = 2 · FAR budget = {args.far_budget:.0%}  "
    )
    out.append(
        f"Development folds: test years {first_test}–{last_test}"
        " (expanding window, val ≥ 8 positives)  "
    )
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
        "| # | Start | End | Wks"
        " | Level hit | Level w-in | z52 hit | z52 w-in | Persist† | Always |"
    )
    out.append(
        "|---|-------|-----|-----|-----------|-----------|---------|---------|---------|--------|"
    )
    for i, (rl, rz, rp, ra) in enumerate(
        zip(ep_res_level2, ep_res_z52_2, ep_res_pers2, ep_res_af2), 1
    ):
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
        out.append(
            f"| {mg} | {len(eps)} | {hl} | {hl / len(eps):.3f} | {hz} | {hz / len(eps):.3f} |"
        )
    out.append("")
    out.append("---")
    out.append("")
    out.append("## VIX level inside dev episodes")
    out.append("")
    out.append(
        "Episodes with median VIX < 20 cannot be caught by any VIX threshold rule "
        "(marked **low-VIX**)."
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
        "|---|----------|-------------|----------------"
        "|-----------|-------------|---------|--------|"
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

    out_path = _REPORTS_DIR / f"{today_str}-baselines.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out), encoding="utf-8")
    rel = out_path.relative_to(out_path.parent.parent)
    print(f"baselines: {len(ep2_dev)} dev episodes → {rel}")
    return 0


# ── select ────────────────────────────────────────────────────────────────────


def _cmd_select(args: argparse.Namespace) -> int:
    from pathlib import Path

    from sqlalchemy.orm import Session

    from .db.engine import create_engine
    from .selection import N_BOOT, run_selection

    out_dir = Path(args.out_dir) if args.out_dir else _REPORTS_DIR
    n_boot = 0 if args.no_boot else (args.n_boot or N_BOOT)

    try:
        engine = create_engine(url=args.database_url)
        with Session(engine) as session:
            _require_db_ready(engine)
            return run_selection(
                session=session,
                out_dir=out_dir,
                label_set=args.label_set,
                first_test_year=args.first_test_year,
                last_test_year=args.last_test_year,
                n_boot=n_boot,
                csv_path=Path(args.csv) if args.csv else None,
                skip_bootstrap=args.no_boot,
            )
    except SystemExit:
        raise
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mad-model",
        description="Market Anomaly Detector — model management commands.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ── episodes ─────────────────────────────────────────────────────────────
    p = subparsers.add_parser(
        "episodes",
        help=(
            "Extract anomaly episodes from the labels table and write "
            "docs/reports/episodes-<label_set>.md."
        ),
    )
    p.add_argument(
        "--label-set",
        default="ews_v1",
        metavar="NAME",
        help="Label set to read from the database (default: ews_v1).",
    )
    p.add_argument(
        "--merge-gap",
        type=int,
        default=2,
        metavar="N",
        help="Maximum calm weeks between runs to merge into one episode (default: 2).",
    )
    p.add_argument("--database-url", default=None, metavar="URL")

    # ── baselines ─────────────────────────────────────────────────────────────
    pb = subparsers.add_parser(
        "baselines",
        help=(
            "Run VIX level and VIX z52 baselines on development folds (labels + VIX"
            " from DB) and write docs/reports/<date>-baselines.md."
        ),
    )
    pb.add_argument(
        "--label-set",
        default="ews_v1",
        metavar="NAME",
        help="Label set to read from the database (default: ews_v1).",
    )
    pb.add_argument(
        "--horizon",
        type=int,
        default=4,
        metavar="H",
        help="Predict horizon in weeks (default: 4).",
    )
    pb.add_argument(
        "--far-budget",
        type=float,
        default=0.05,
        metavar="B",
        help="False-alarm rate budget for threshold selection (default: 0.05).",
    )
    pb.add_argument(
        "--first-test-year",
        type=int,
        default=2008,
        metavar="YEAR",
        help="First test year in the development folds (default: 2008).",
    )
    pb.add_argument(
        "--last-test-year",
        type=int,
        default=2017,
        metavar="YEAR",
        help="Last test year in the development folds (default: 2017).",
    )
    pb.add_argument("--database-url", default=None, metavar="URL")

    # ── select ────────────────────────────────────────────────────────────────
    ps = subparsers.add_parser(
        "select",
        help=(
            "Run candidate models on development folds and select the best one. "
            "Writes docs/reports/<date>-model-selection.md. Requires a populated DB."
        ),
    )
    ps.add_argument("--label-set", default="ews_v1", metavar="NAME")
    ps.add_argument("--database-url", default=None, metavar="URL")
    ps.add_argument(
        "--out-dir",
        default=None,
        metavar="DIR",
        help="Directory to write the report (default: docs/reports/).",
    )
    ps.add_argument(
        "--no-boot",
        action="store_true",
        help="Skip year-block bootstrap (faster, no confidence intervals).",
    )
    ps.add_argument(
        "--n-boot",
        type=int,
        default=None,
        metavar="N",
        help="Bootstrap resamples (default: 2000).",
    )
    ps.add_argument(
        "--first-test-year",
        type=int,
        default=2008,
        metavar="YEAR",
    )
    ps.add_argument(
        "--last-test-year",
        type=int,
        default=2017,
        metavar="YEAR",
    )
    ps.add_argument(
        "--csv",
        default=None,
        metavar="PATH",
        help="Path to old EWS CSV for evaluating old baselines (optional).",
    )

    args = parser.parse_args()
    if args.command == "episodes":
        sys.exit(_cmd_episodes(args))
    elif args.command == "baselines":
        sys.exit(_cmd_baselines(args))
    elif args.command == "select":
        sys.exit(_cmd_select(args))
    else:
        parser.print_help()
        sys.exit(1)
