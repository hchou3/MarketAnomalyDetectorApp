"""Phase 2 Step 2.5 — final confirmation on 2018–2021, run once.

The selected model (hgb) is evaluated on the held-out confirmation window.
A frozen configuration dataclass hashes all hyperparameters; the run-once
guard refuses re-runs unless --force is given with an explicit reason.

CLI entry: mad-model confirm --csv <path>
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, fields
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .eval_harness import (
    KNOWN_UNLABELED,
    always_flag,
    detect_event_results,
    persistence,
    predict_event_results,
    row_metrics,
)
from .selection import (
    HORIZON,
    N_BOOT,
    CandidateResult,
    FoldResult,
    _evaluate_old_baselines,
    _evaluate_vix_baselines,
    build_export_df,
    compute_pooled_metrics,
    evaluate_candidate,
    make_hgb_pipeline,
    year_block_bootstrap,
)

CONFIRM_FIRST_YEAR = 2018
CONFIRM_LAST_YEAR = 2021
CONFIRM_LAST_LABEL = date(2021, 4, 20)

# Development numbers (from the committed selection report) for the dev/confirm table.
_DEV_HGB_EVENT_F1 = 0.527
_DEV_HGB_ROW_F1 = 0.429
_DEV_HGB_PR_AUC = 0.571
_DEV_HGB_BRIER = 0.148
_DEV_HGB_LOG_LOSS: float | None = None  # not in selection report summary
_DEV_VIX_EVENT_F1 = 0.511
_DEV_VIX_ROW_F1 = 0.556
_DEV_VIX_PR_AUC = 0.654
_DEV_VIX_BRIER = 0.161


# ── Frozen configuration ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class ConfirmConfig:
    """All hyperparameters and policy choices that define the confirmation run.

    Changing any field changes config_hash, which triggers the run-once guard.
    """

    model: str = "hgb"
    hgb_max_depth: int = 3
    hgb_learning_rate: float = 0.05
    hgb_max_iter: int = 200
    hgb_l2_regularization: float = 1.0
    hgb_random_state: int = 42
    feature_spec: str = "SPEC_V1"
    calibration: str = "platt_raw_margin"
    far_budget: float = 0.05
    horizon: int = 4
    merge_gap: int = 2
    feature_rule: str = "core_ok_vix_sp500_us10y"
    first_test_year: int = CONFIRM_FIRST_YEAR
    last_test_year: int = CONFIRM_LAST_YEAR


FROZEN_CONFIG = ConfirmConfig()


def config_hash(cfg: ConfirmConfig = FROZEN_CONFIG) -> str:
    """SHA-256 of the canonical JSON representation of *cfg*."""
    d = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
    payload = json.dumps(d, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _git_sha() -> str:
    """Return the current HEAD SHA (short), or 'unknown' on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


# ── Run-once guard ────────────────────────────────────────────────────────────


def run_once_check(
    out_dir: Path,
    cfg_hash: str,
    force: bool,
    reason: str,
) -> None:
    """Abort (raise SystemExit) if a report with *cfg_hash* already exists.

    If *force* is True and *reason* is non-empty, print a warning and continue.
    """
    short = cfg_hash[:8]
    existing = list(out_dir.glob(f"*-final-confirmation-{short}.md"))
    if not existing:
        return
    if force and reason:
        print(
            f"WARNING: confirm re-run forced. Reason: {reason}. "
            f"Existing report(s): {[p.name for p in existing]}"
        )
        return
    raise SystemExit(
        f"ERROR: A final-confirmation report for config hash {short} already exists:\n"
        + "\n".join(f"  {p}" for p in existing)
        + "\nRe-running would overwrite a committed result. "
        "Use --force --reason '<text>' to override."
    )


# ── Per-fold evaluation (reuses evaluate_candidate with _guard=False) ────────


def _evaluate_hgb_confirm(
    X_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    vix_close: np.ndarray,
    folds,
    ku_mask: np.ndarray,
) -> CandidateResult:
    """Run hgb on confirmation folds (2018–2021) using the frozen config."""
    return evaluate_candidate(
        "hgb",
        make_hgb_pipeline,
        X_df,
        labels_df,
        folds,
        ku_mask,
        _guard=False,
    )


# ── Persistence + always-flag folds ──────────────────────────────────────────


def _evaluate_simple_baselines(
    labels_df: pd.DataFrame,
    folds,
) -> dict[str, CandidateResult]:
    """Evaluate persistence and always-flag on confirmation folds."""
    from .eval_harness import FoldSpec  # noqa: F401

    all_weeks: list[date] = list(labels_df["week"])
    detect_arr = labels_df["value"].values.astype(int)

    from .selection import OnnxGateResult

    _no_gate = OnnxGateResult(passed=False, max_diff=None, error="Simple baseline — no ONNX export")

    pers_folds: list[FoldResult] = []
    af_folds: list[FoldResult] = []

    pers_all = persistence(detect_arr)

    for fold in folds:
        test_weeks = all_weeks[fold.test_lo : fold.test_hi]
        y_test = detect_arr[fold.test_lo : fold.test_hi]
        pers_alerts = pers_all[fold.test_lo : fold.test_hi]
        af_alerts = always_flag(len(y_test))

        pers_folds.append(
            FoldResult(
                fold_year=fold.test_year,
                head="detect",
                weeks=test_weeks,
                true_labels=y_test,
                proba=pers_alerts.astype(float),
                alerts=pers_alerts,
                threshold=0.5,
                platt_a=1.0,
                platt_b=0.0,
            )
        )
        af_folds.append(
            FoldResult(
                fold_year=fold.test_year,
                head="detect",
                weeks=test_weeks,
                true_labels=y_test,
                proba=af_alerts.astype(float),
                alerts=af_alerts,
                threshold=0.5,
                platt_a=1.0,
                platt_b=0.0,
            )
        )

    return {
        "persistence": CandidateResult(
            name="persistence", detect_folds=pers_folds, predict_folds=[], onnx_gate=_no_gate
        ),
        "always_flag": CandidateResult(
            name="always_flag", detect_folds=af_folds, predict_folds=[], onnx_gate=_no_gate
        ),
    }


# ── Export predictions ────────────────────────────────────────────────────────


def build_confirm_export_df(
    hgb_result: CandidateResult,
    vix_result: CandidateResult,
    all_weeks: list[date],
    vix_close: np.ndarray,
    ku_mask: np.ndarray,
    episodes,
) -> pd.DataFrame:
    """Assemble per-confirm-week DataFrame, same columns as dev_predictions.csv."""
    return build_export_df(hgb_result, vix_result, all_weeks, vix_close, ku_mask, episodes)


# ── Report writer ─────────────────────────────────────────────────────────────


def _fmt_ci(ci: list | None) -> str:
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "—"


def _pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "0/0"


def write_confirm_report(
    out_path: Path,
    hgb_result: CandidateResult,
    vix_result: CandidateResult,
    baselines: dict[str, CandidateResult],
    pooled: dict[str, dict],
    boot: dict[str, dict],
    episodes,
    confirm_weeks: list[date],
    ku_mask: np.ndarray,
    vix_close: np.ndarray,
    cfg_hash: str,
    git_sha: str,
    force_reason: str,
    n_boot: int,
    label_set: str,
) -> None:
    today = date.today().isoformat()
    short_hash = cfg_hash[:8]
    all_results = {"hgb": hgb_result, "vix_level": vix_result, **baselines}

    # Calm confirm years (years with no labeled episode)
    ep_years: set[int] = set()
    for ep in episodes:
        d = ep.start
        while d <= ep.end:
            ep_years.add(d.year)
            d += timedelta(weeks=1)
    confirm_year_range = range(CONFIRM_FIRST_YEAR, CONFIRM_LAST_YEAR + 1)

    lines: list[str] = [
        f"# Final Confirmation Report — {label_set} (2018–2021)",
        "",
        f"Generated: {today}  ",
        f"Config hash: `{short_hash}` (full: `{cfg_hash}`)  ",
        f"Git SHA: `{git_sha}`  ",
        f"Label set: `{label_set}` · Horizon h = {HORIZON} · Merge gap = 2 · FAR budget = 5%  ",
        f"Confirmation folds: test years {CONFIRM_FIRST_YEAR}–{CONFIRM_LAST_YEAR}"
        f" (last label: {CONFIRM_LAST_LABEL})  ",
        "",
    ]

    if force_reason:
        lines += [
            f"> **Forced re-run.** Reason: {force_reason}",
            "",
        ]

    lines += [
        "> **Run-once result.** This report is committed unedited and not rerun.",
        "> The 2018–2021 window was held out throughout development.",
        "",
        "---",
        "",
        "## Summary table",
        "",
        "| Model | Event F1 | Episodes caught | Row F1 | PR AUC | Brier | Log loss"
        " | FA total | FA per year |",
        "|-------|---------|----------------|--------|--------|-------|----------|"
        "---------|-------------|",
    ]

    model_order = [
        "hgb",
        "vix_level",
        "vix_z52",
        "always_flag",
        "persistence",
        "old_ensemble",
        "old_ensemble_native05",
    ]
    for nm in model_order:
        if nm not in pooled:
            continue
        m = pooled[nm]
        pm = m.get("prob_metrics") or {}
        pr_auc = f"{pm.get('pr_auc', float('nan')):.3f}" if pm else "—"
        brier = f"{pm.get('brier', float('nan')):.3f}" if pm else "—"
        ll = f"{pm.get('log_loss', float('nan')):.3f}" if pm else "—"
        hits = f"{m['episode_hits']}/{m['episodes_total']}"
        row_f1 = m["row"]["f1"]
        fa_total = m.get("fa_total_runs", "—")
        n_years = len(confirm_year_range)
        fa_per_yr = (
            f"{m.get('fa_total_runs', 0) / n_years:.2f}"
            if isinstance(m.get("fa_total_runs"), int)
            else "—"
        )
        lines.append(
            f"| {nm} | {m['event_f1']:.3f} | {hits} | {row_f1:.3f}"
            f" | {pr_auc} | {brier} | {ll} | {fa_total} | {fa_per_yr} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Per-episode hit table",
        "",
        "Detect head (merge_gap=2, grace=±2 weeks). First-alert week shown for hgb and vix_level.",
        "",
    ]

    if episodes:
        ep_cols = ["hgb", "vix_level"]
        header = ["#", "Start", "End", "Wks", "Historical match"]
        for nm in ep_cols:
            header += [f"{nm} hit", f"{nm} first alert"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for i, ep in enumerate(episodes, 1):
            from .model_cli import _historical_match

            match = _historical_match(ep.start, ep.end)
            row_parts = [str(i), str(ep.start), str(ep.end), str(ep.length), match]
            for nm in ep_cols:
                cr = all_results.get(nm)
                if not cr or not cr.detect_folds:
                    row_parts += ["—", "—"]
                    continue
                all_alerts = {
                    w for fr in cr.detect_folds for w, a in zip(fr.weeks, fr.alerts) if a == 1
                }
                ep_res = detect_event_results([ep], all_alerts)
                r = ep_res[0]
                if r.hit:
                    row_parts += ["Y", str(r.first_alert_week)]
                else:
                    row_parts += ["N", "—"]
            lines.append("| " + " | ".join(row_parts) + " |")

    # ── Predict head ─────────────────────────────────────────────────────────
    lines += [
        "",
        "---",
        "",
        f"## Predict head (h={HORIZON} weeks)",
        "",
        "Row metrics evaluated on calm-at-t weeks only (detect==0 with valid predict target).",
        "",
        "| Model | Precision | Recall | F1 |",
        "|-------|-----------|--------|----|",
    ]

    all_detect_weeks: dict[date, int] = {}
    for cr in all_results.values():
        for fr in cr.detect_folds:
            for w, lbl in zip(fr.weeks, fr.true_labels):
                all_detect_weeks[w] = int(lbl)

    pred_names = [nm for nm in ["hgb", "vix_level"] if nm in all_results]
    for nm in pred_names:
        cr = all_results[nm]
        if not cr.predict_folds:
            lines.append(f"| {nm} | — | — | — |")
            continue
        yt = np.concatenate([fr.true_labels for fr in cr.predict_folds])
        ya = np.concatenate([fr.alerts for fr in cr.predict_folds])
        rm = row_metrics(yt, ya)
        lines.append(f"| {nm} | {rm['precision']:.3f} | {rm['recall']:.3f} | {rm['f1']:.3f} |")

    lines += [
        "",
        "### Advance warning per episode (predict-head alert ≤ h weeks before onset)",
        "",
    ]
    if episodes:
        aw_header = ["#", "Start", "Wks"] + pred_names
        lines.append("| " + " | ".join(aw_header) + " |")
        lines.append("|" + "---|" * len(aw_header))
        for i, ep in enumerate(episodes, 1):
            row_parts = [str(i), str(ep.start), str(ep.length)]
            for nm in pred_names:
                cr = all_results.get(nm)
                if not cr or not cr.predict_folds:
                    row_parts.append("—")
                    continue
                calm_alerts = {
                    w
                    for fr in cr.predict_folds
                    for w, a in zip(fr.weeks, fr.alerts)
                    if a == 1 and all_detect_weeks.get(w, 1) == 0
                }
                ep_res = predict_event_results([ep], calm_alerts, HORIZON)
                r = ep_res[0]
                row_parts.append(
                    f"Y (+{r.weeks_into_episode}w)" if r.hit and r.weeks_into_episode else "N"
                )
            lines.append("| " + " | ".join(row_parts) + " |")

    # ── Known-unlabeled windows ───────────────────────────────────────────────
    lines += [
        "",
        "---",
        "",
        "## Known-unlabeled windows in confirmation period",
        "",
        "Alerts in these windows are reported but excluded from false-alarm counts and thresholds.",
        "",
        "| Event | Window | hgb alerts | vix_level alerts |",
        "|-------|--------|-----------|-----------------|",
    ]

    hgb_alert_set = {
        w for fr in hgb_result.detect_folds for w, a in zip(fr.weeks, fr.alerts) if a == 1
    }
    vix_alert_set = {
        w for fr in vix_result.detect_folds for w, a in zip(fr.weeks, fr.alerts) if a == 1
    }

    confirm_ku = [(ks, ke, kn) for ks, ke, kn in KNOWN_UNLABELED if ke >= date(2018, 1, 1)]
    for ks, ke, kname in confirm_ku:
        hc = sum(1 for w in confirm_weeks if ks <= w <= ke and w in hgb_alert_set)
        vc = sum(1 for w in confirm_weeks if ks <= w <= ke and w in vix_alert_set)
        lines.append(f"| {kname} | {ks} – {ke} | {hc} | {vc} |")

    # ── Case studies ──────────────────────────────────────────────────────────
    lines += [
        "",
        "---",
        "",
        "## Case studies",
        "",
        "### 2018 Q4 (global equity sell-off)",
        "",
        "| Week | hgb prob | hgb alert | vix_level alert | Label |",
        "|------|---------|-----------|----------------|-------|",
    ]

    hgb_by_week: dict[date, dict] = {}
    for fr in hgb_result.detect_folds:
        for w, p, a, lbl in zip(fr.weeks, fr.proba, fr.alerts, fr.true_labels):
            hgb_by_week[w] = {"prob": float(p), "alert": int(a), "label": int(lbl)}
    vix_by_week: dict[date, dict] = {}
    for fr in vix_result.detect_folds:
        for w, a in zip(fr.weeks, fr.alerts):
            vix_by_week[w] = {"alert": int(a)}

    for w in sorted(confirm_weeks):
        if not (date(2018, 10, 1) <= w <= date(2018, 12, 31)):
            continue
        h = hgb_by_week.get(w)
        v = vix_by_week.get(w)
        prob = f"{h['prob']:.4f}" if h else "—"
        ha = str(h["alert"]) if h else "—"
        va = str(v["alert"]) if v else "—"
        lbl = str(h["label"]) if h else "—"
        lines.append(f"| {w} | {prob} | {ha} | {va} | {lbl} |")

    lines += [
        "",
        "### 2020 (COVID-19)",
        "",
        "| Week | hgb prob | hgb alert | vix_level alert | Label |",
        "|------|---------|-----------|----------------|-------|",
    ]

    for w in sorted(confirm_weeks):
        if w.year != 2020:
            continue
        h = hgb_by_week.get(w)
        v = vix_by_week.get(w)
        prob = f"{h['prob']:.4f}" if h else "—"
        ha = str(h["alert"]) if h else "—"
        va = str(v["alert"]) if v else "—"
        lbl = str(h["label"]) if h else "—"
        lines.append(f"| {w} | {prob} | {ha} | {va} | {lbl} |")

    # ── Bootstrap ─────────────────────────────────────────────────────────────
    lines += [
        "",
        "---",
        "",
        f"## Bootstrap intervals ({n_boot} resamples of test years)",
        "",
        "> **Note:** With only 4 test years, year-block bootstrap intervals are not"
        " meaningful — each resample is one of 4^4 = 256 year combinations and the"
        " distribution is discrete. The per-episode table above is the main evidence.",
        "",
        "| Model | Event F1 95% CI | Row F1 95% CI | PR AUC 95% CI | Brier 95% CI |",
        "|-------|----------------|--------------|--------------|-------------|",
    ]

    for nm in model_order:
        if nm not in boot:
            continue
        b = boot[nm]
        lines.append(
            f"| {nm} | {_fmt_ci(b.get('event_f1_ci95'))} | {_fmt_ci(b.get('row_f1_ci95'))}"
            f" | {_fmt_ci(b.get('pr_auc_ci95'))} | {_fmt_ci(b.get('brier_ci95'))} |"
        )

    # ── Development vs confirmation table ──────────────────────────────────────
    def _m(key: str, m: dict | None) -> str:
        if m is None:
            return "—"
        pm = m.get("prob_metrics") or {}
        if key == "event_f1":
            return f"{m.get('event_f1', float('nan')):.3f}"
        if key == "row_f1":
            return f"{m['row']['f1']:.3f}"
        if key == "pr_auc":
            return f"{pm.get('pr_auc', float('nan')):.3f}" if pm else "—"
        if key == "brier":
            return f"{pm.get('brier', float('nan')):.3f}" if pm else "—"
        return "—"

    hgb_conf = pooled.get("hgb")
    vix_conf = pooled.get("vix_level")

    lines += [
        "",
        "---",
        "",
        "## Development vs confirmation",
        "",
        "Dev numbers are from the committed model-selection report (2008–2017).",
        "",
        "| Model | Period | Event F1 | Row F1 | PR AUC | Brier |",
        "|-------|--------|---------|--------|--------|-------|",
        f"| hgb | dev (2008–2017) | {_DEV_HGB_EVENT_F1:.3f} | {_DEV_HGB_ROW_F1:.3f}"
        f" | {_DEV_HGB_PR_AUC:.3f} | {_DEV_HGB_BRIER:.3f} |",
        f"| hgb | confirm (2018–2021) | {_m('event_f1', hgb_conf)} | {_m('row_f1', hgb_conf)}"
        f" | {_m('pr_auc', hgb_conf)} | {_m('brier', hgb_conf)} |",
        f"| vix_level | dev (2008–2017) | {_DEV_VIX_EVENT_F1:.3f} | {_DEV_VIX_ROW_F1:.3f}"
        f" | {_DEV_VIX_PR_AUC:.3f} | {_DEV_VIX_BRIER:.3f} |",
        f"| vix_level | confirm (2018–2021) | {_m('event_f1', vix_conf)} | {_m('row_f1', vix_conf)}"
        f" | {_m('pr_auc', vix_conf)} | {_m('brier', vix_conf)} |",
    ]

    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ── CLI entry point ───────────────────────────────────────────────────────────


def run_confirm(
    session,
    out_dir: Path,
    label_set: str = "ews_v1",
    first_test_year: int = CONFIRM_FIRST_YEAR,
    last_test_year: int = CONFIRM_LAST_YEAR,
    n_boot: int = N_BOOT,
    csv_path: Path | None = None,
    skip_bootstrap: bool = False,
    force: bool = False,
    force_reason: str = "",
) -> int:
    """Load data, run hgb + baselines on confirmation folds, write report."""
    from .episodes import extract_episodes
    from .eval_harness import make_folds_v2
    from .features.build import build_from_db, weekly_close
    from .features.spec import SPEC_V1
    from .labels import load_labels

    cfg = FROZEN_CONFIG
    cfg_hash_val = config_hash(cfg)
    git_sha = _git_sha()

    # ── Run-once guard ────────────────────────────────────────────────────────
    run_once_check(out_dir, cfg_hash_val, force, force_reason)

    # ── Load data ─────────────────────────────────────────────────────────────
    labels_df = load_labels(session, label_set=label_set)
    if labels_df.empty:
        print(f"ERROR: no labels for {label_set!r}")
        return 1

    weeks: list[date] = list(labels_df["week"])
    feature_df = build_from_db(session, SPEC_V1, weeks, as_of=None)
    vix_close = weekly_close(session, "vix", weeks)

    ku_mask = np.zeros(len(weeks), dtype=bool)
    for ks, ke, _ in KNOWN_UNLABELED:
        for i, w in enumerate(weeks):
            if ks <= w <= ke:
                ku_mask[i] = True

    # ── Confirmation folds ────────────────────────────────────────────────────
    confirm_folds = make_folds_v2(
        weeks,
        labels_df["value"].values.astype(int),
        first_test_year=first_test_year,
        last_test_year=last_test_year,
    )

    # Confirmation episodes and weeks
    ep_all = extract_episodes(labels_df, merge_gap=cfg.merge_gap)
    confirm_start = date(first_test_year, 1, 1)
    confirm_end = date(last_test_year, 12, 31)
    episodes = [e for e in ep_all if confirm_start <= e.start <= confirm_end]
    confirm_weeks = [w for w in weeks if confirm_start <= w <= confirm_end]

    calm_confirm_years = [
        y
        for y in range(first_test_year, last_test_year + 1)
        if not any(e.start <= date(y, 12, 31) and e.end >= date(y, 1, 1) for e in episodes)
    ]

    # ── Evaluate hgb ─────────────────────────────────────────────────────────
    print("confirm: evaluating hgb ...")
    hgb_result = _evaluate_hgb_confirm(feature_df, labels_df, vix_close, confirm_folds, ku_mask)

    # ── VIX baselines ─────────────────────────────────────────────────────────
    print("confirm: evaluating vix baselines ...")
    vix_baselines = _evaluate_vix_baselines(
        labels_df, confirm_folds, vix_close, ku_mask, _guard=False
    )

    # ── Simple baselines (persistence, always-flag) ───────────────────────────
    simple_baselines = _evaluate_simple_baselines(labels_df, confirm_folds)

    # ── Old baselines (optional, requires CSV) ────────────────────────────────
    old_baselines: dict[str, CandidateResult] = {}
    if csv_path is not None and csv_path.exists():
        print("confirm: evaluating old ensemble ...")
        old_baselines = _evaluate_old_baselines(
            labels_df, confirm_folds, csv_path, ku_mask, _guard=False
        )
    else:
        print("confirm: old CSV not provided; old_ensemble skipped.")

    all_baselines = {**vix_baselines, **simple_baselines, **old_baselines}

    # ── Pooled metrics ────────────────────────────────────────────────────────
    pooled: dict[str, dict] = {}
    pooled["hgb"] = compute_pooled_metrics(
        hgb_result.detect_folds, episodes, confirm_weeks, calm_confirm_years
    )
    for nm, cr in all_baselines.items():
        if cr.detect_folds:
            pooled[nm] = compute_pooled_metrics(
                cr.detect_folds, episodes, confirm_weeks, calm_confirm_years
            )

    # ── Bootstrap ─────────────────────────────────────────────────────────────
    boot: dict[str, dict] = {}
    _empty_boot = {
        "n_boot": 0,
        "event_f1_ci95": None,
        "event_f1_se": 0.0,
        "row_f1_ci95": None,
        "pr_auc_ci95": None,
        "brier_ci95": None,
        "log_loss_ci95": None,
        "_boot_event_f1": [],
        "_boot_row_f1": [],
    }

    if not skip_bootstrap:
        print("confirm: bootstrapping hgb ...")
        boot["hgb"] = year_block_bootstrap(
            hgb_result.detect_folds, episodes, confirm_weeks, n_boot=n_boot
        )
        for nm, cr in all_baselines.items():
            if cr.detect_folds:
                boot[nm] = year_block_bootstrap(
                    cr.detect_folds, episodes, confirm_weeks, n_boot=n_boot
                )
    else:
        boot["hgb"] = _empty_boot
        for nm in all_baselines:
            boot[nm] = _empty_boot

    # ── Write report ──────────────────────────────────────────────────────────
    short_hash = cfg_hash_val[:8]
    today = date.today().isoformat()
    out_path = out_dir / f"{today}-final-confirmation-{short_hash}.md"

    _no_vix_gate = hgb_result.onnx_gate
    vix_result = vix_baselines.get(
        "vix_level",
        CandidateResult(
            name="vix_level", detect_folds=[], predict_folds=[], onnx_gate=_no_vix_gate
        ),
    )

    write_confirm_report(
        out_path=out_path,
        hgb_result=hgb_result,
        vix_result=vix_result,
        baselines=all_baselines,
        pooled=pooled,
        boot=boot,
        episodes=episodes,
        confirm_weeks=confirm_weeks,
        ku_mask=ku_mask,
        vix_close=vix_close,
        cfg_hash=cfg_hash_val,
        git_sha=git_sha,
        force_reason=force_reason,
        n_boot=0 if skip_bootstrap else n_boot,
        label_set=label_set,
    )
    print(f"confirm: -> {out_path}")

    # ── Export predictions CSV ────────────────────────────────────────────────
    artifacts_dir = out_dir.parent.parent / "artifacts"
    confirm_csv = artifacts_dir / "confirm_predictions.csv"
    try:
        confirm_df = build_confirm_export_df(
            hgb_result, vix_result, weeks, vix_close, ku_mask, episodes
        )
        confirm_csv.parent.mkdir(parents=True, exist_ok=True)
        confirm_df.to_csv(confirm_csv, index=False)
        print(f"confirm: predictions -> {confirm_csv}")
    except Exception as exc:
        print(f"confirm: WARNING predictions export failed: {exc}")

    n_ep = pooled["hgb"]["episode_hits"]
    n_tot = pooled["hgb"]["episodes_total"]
    ef1 = pooled["hgb"]["event_f1"]
    print(f"confirm: hgb event F1={ef1:.3f}, episodes caught={n_ep}/{n_tot}")
    return 0
