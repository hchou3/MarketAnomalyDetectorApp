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

_DEFAULT_CSV = Path(__file__).parent.parent.parent / "data" / "raw" / "ews_weekly_2000_2021.csv"


def _cmd_baselines(args: argparse.Namespace) -> int:
    import math
    from datetime import timedelta

    import numpy as np
    import pandas as pd

    from .episodes import extract_episodes
    from .eval_harness import (
        KNOWN_UNLABELED,
        detect_event_results,
        false_alarm_runs,
        fit_vix_threshold,
        make_folds_v2,
        persistence,
        predict_event_results,
        row_metrics,
        vix_rule,
    )
    from .labels import build_targets

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(csv_path, parse_dates=["Data"])
    df = df.sort_values("Data").reset_index(drop=True)
    weeks: list[date] = [d.date() for d in df["Data"]]
    detect = df["Y"].values.astype(int)

    # VIX 156-week z-score of 1-week log return
    log_vix = np.log(df["VIX"].values)
    vix_lr = np.empty_like(log_vix)
    vix_lr[0] = np.nan
    vix_lr[1:] = np.diff(log_vix)
    vix_z = np.full(len(vix_lr), np.nan)
    for i in range(len(vix_lr)):
        w = vix_lr[max(0, i - 155) : i + 1]
        w = w[~np.isnan(w)]
        if len(w) >= 104 and not np.isnan(vix_lr[i]):
            std = w.std(ddof=1)
            vix_z[i] = (vix_lr[i] - w.mean()) / std if std > 0 else 0.0

    labels_df = pd.DataFrame({"week": weeks, "value": detect})
    targets = build_targets(labels_df, horizon=args.horizon)
    predict_labels = targets["predict"].values

    ep2 = extract_episodes(labels_df, merge_gap=2)
    ep0 = extract_episodes(labels_df, merge_gap=0)
    ep4 = extract_episodes(labels_df, merge_gap=4)

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

    rows_detect: list[dict] = []
    rows_predict: list[dict] = []
    for fold in folds:
        val_vix_z = vix_z[fold.val_lo : fold.test_lo]
        val_detect = detect[fold.val_lo : fold.test_lo]
        val_ku = ku_mask[fold.val_lo : fold.test_lo]
        thr = fit_vix_threshold(
            val_vix_z, val_detect, far_budget=args.far_budget, excluded_mask=val_ku
        )

        t_detect = detect[fold.test_lo : fold.test_hi]
        t_vix_z = vix_z[fold.test_lo : fold.test_hi]
        t_pred = predict_labels[fold.test_lo : fold.test_hi]
        t_weeks = weeks[fold.test_lo : fold.test_hi]

        per = persistence(t_detect)
        vr = vix_rule(t_vix_z, thr)

        for k, (wk, dl, pl) in enumerate(zip(t_weeks, t_detect, t_pred)):
            rows_detect.append(
                {
                    "week": wk,
                    "detect": int(dl),
                    "always": 1,
                    "persist": int(per[k]),
                    "vix": int(vr[k]),
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
                        "vix": int(vr[k]),
                        "fold": fold.test_year,
                    }
                )

    det_df = pd.DataFrame(rows_detect)
    pred_df = pd.DataFrame(rows_predict)
    calm_pred_df = pred_df[pred_df["detect"] == 0].copy()
    calm_at_t_pos = int((calm_pred_df["predict"] == 1).sum())

    baselines = ["always", "persist", "vix"]
    bnames = {
        "always": "Always-flag",
        "persist": "Persistence (prev-week label)",
        "vix": "VIX z-score rule",
    }
    detect_metrics = {b: row_metrics(det_df["detect"], det_df[b]) for b in baselines}
    predict_metrics = {b: row_metrics(calm_pred_df["predict"], calm_pred_df[b]) for b in baselines}

    first_test = args.first_test_year
    last_test = args.last_test_year or max(w.year for w in weeks)
    dev_start = date(first_test, 1, 1)
    dev_end = date(last_test, 12, 31)

    ep2_dev = [e for e in ep2 if dev_start <= e.start <= dev_end]
    ep0_dev = [e for e in ep0 if dev_start <= e.start <= dev_end]
    ep4_dev = [e for e in ep4 if dev_start <= e.start <= dev_end]

    alert_set = set(det_df[det_df["vix"] == 1]["week"])
    persist_set = set(det_df[det_df["persist"] == 1]["week"])

    ep_res_vix2 = detect_event_results(ep2_dev, alert_set)
    ep_res_pers2 = detect_event_results(ep2_dev, persist_set)
    ep_res_af2 = detect_event_results(ep2_dev, set(det_df["week"]))
    ep_res_vix0 = detect_event_results(ep0_dev, alert_set)
    ep_res_vix4 = detect_event_results(ep4_dev, alert_set)

    hit_vix2 = sum(r.hit for r in ep_res_vix2)
    hit_pers2 = sum(r.hit for r in ep_res_pers2)
    hit_af2 = sum(r.hit for r in ep_res_af2)
    hit_vix0 = sum(r.hit for r in ep_res_vix0)
    hit_vix4 = sum(r.hit for r in ep_res_vix4)

    calm_set_vix = set(det_df[(det_df["vix"] == 1) & (det_df["detect"] == 0)]["week"])
    calm_set_pers = set(det_df[(det_df["persist"] == 1) & (det_df["detect"] == 0)]["week"])
    calm_set_always = set(det_df[det_df["detect"] == 0]["week"])

    pred_res_vix = predict_event_results(ep2_dev, calm_set_vix, horizon=args.horizon)
    pred_res_pers = predict_event_results(ep2_dev, calm_set_pers, horizon=args.horizon)
    pred_res_af = predict_event_results(ep2_dev, calm_set_always, horizon=args.horizon)

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
    calm_dev_years = [y for y in range(first_test, last_test + 1) if y not in ep2_dev_years]
    fa_vix_calm = [r for r in fa_vix_real if r.year in calm_dev_years]
    fa_pers_calm = [r for r in fa_pers_real if r.year in calm_dev_years]

    def pct(n: int, d: int) -> str:
        return f"{n}/{d} ({100 * n // d}%)" if d else "0/0"

    today_str = date.today().isoformat()
    out: list[str] = []
    out.append(f"# Baselines Report — ews_v1 Development Folds ({first_test}–{last_test})")
    out.append("")
    out.append(f"Generated: {today_str}  ")
    out.append(f"Data source: `{csv_path.name}`  ")
    out.append(
        f"Label set: `ews_v1` · Horizon h = {args.horizon}"
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
            f"| {bnames[b]} | {m['precision']:.3f} | {m['recall']:.3f}"
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
        out.append(
            f"| {i} | {ep.start} | {ep.end} | {ep.length}"
            f" | {'Y' if rv.hit else 'N'} | {rv.weeks_into_episode if rv.hit else '-'}"
            f" | {'Y' if rp.hit else 'N'} | {'Y' if ra.hit else 'N'} |"
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
            f"| {bnames[b]} | {m['precision']:.3f} | {m['recall']:.3f}"
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
        out.append(
            f"| {i} | {ep.start} | {'Y' if rv.hit else 'N'}"
            f" | {rv.weeks_into_episode if rv.hit else '-'}"
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
    out.append("| Baseline | Total FA runs | FA in calm years | FA runs in known-unlabeled |")
    out.append("|----------|--------------|-----------------|---------------------------|")
    ku_vix = sum(1 for r in fa_vix if r.in_known_unlabeled)
    ku_pers = sum(1 for r in fa_pers if r.in_known_unlabeled)
    out.append(f"| VIX rule | {len(fa_vix_real)} | {len(fa_vix_calm)} | {ku_vix} |")
    out.append(f"| Persistence | {len(fa_pers_real)} | {len(fa_pers_calm)} | {ku_pers} |")
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

    out_path = _REPORTS_DIR / f"{today_str}-baselines.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out), encoding="utf-8")
    rel = out_path.relative_to(out_path.parent.parent)
    print(f"baselines: {len(ep2_dev)} dev episodes → {rel}")
    return 0


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
            "Run baselines (always-flag, persistence, VIX rule) on development folds "
            "and write docs/reports/<date>-baselines.md."
        ),
    )
    pb.add_argument(
        "--csv",
        default=str(_DEFAULT_CSV),
        metavar="PATH",
        help="Path to the EWS weekly CSV (default: data/raw/ews_weekly_2000_2021.csv).",
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

    args = parser.parse_args()
    if args.command == "episodes":
        sys.exit(_cmd_episodes(args))
    elif args.command == "baselines":
        sys.exit(_cmd_baselines(args))
    else:
        parser.print_help()
        sys.exit(1)
