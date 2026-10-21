"""Phase 2 Step 2.4b — pre-registered HGB experiment batch.

Six HGB variants evaluated on dev folds 2008–2017:
  hgb_base   — current HGB (reference)
  hgb_vixlvl — + raw VIX close appended as an extra feature
  hgb_oof    — OOF calibration + threshold over the full pre-test history
  hgb_hyst   — hysteresis: alert on when p >= thr; stays on until p < 0.5*thr
  hgb_epw    — episode-proportional sample weights during training
  hgb_all    — vixlvl + oof + hyst + epw combined

All variants: same HGB hyperparameters, Platt (raw margin) calibration, and
selection rule. VIX level is the evaluation bar.
Run once — no post-hoc tuning.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .calibration import PlattCalibrator
from .episodes import Episode
from .eval_harness import (
    KNOWN_UNLABELED,
    FoldSpec,
    choose_threshold,
    detect_event_results,
    false_alarm_runs,
)
from .selection import (
    DEV_FIRST_YEAR,
    DEV_LAST_YEAR,
    FAR_BUDGET,
    GRACE_WEEKS,
    N_BOOT,
    CandidateResult,
    FoldResult,
    OnnxGateResult,
    _core_feature_mask,
    _guard_dev_only,
    _set_njobs_1,
    check_onnx_gate,
    compute_pooled_metrics,
    make_hgb_pipeline,
    year_block_bootstrap,
)

OPERATING_CURVE_BUDGETS = [0.02, 0.05, 0.10]


# ── Dataclasses ───────────────────────────────────────────────────────────────


@dataclass
class HGBVariantConfig:
    name: str
    use_vixlvl: bool = False
    use_oof: bool = False
    use_hyst: bool = False
    use_epw: bool = False
    hyst_factor: float = 0.5


@dataclass
class ValFoldData:
    fold_year: int
    val_proba: np.ndarray
    val_labels: np.ndarray
    val_ku: np.ndarray


VARIANTS: dict[str, HGBVariantConfig] = {
    "hgb_base": HGBVariantConfig("hgb_base"),
    "hgb_vixlvl": HGBVariantConfig("hgb_vixlvl", use_vixlvl=True),
    "hgb_oof": HGBVariantConfig("hgb_oof", use_oof=True),
    "hgb_hyst": HGBVariantConfig("hgb_hyst", use_hyst=True),
    "hgb_epw": HGBVariantConfig("hgb_epw", use_epw=True),
    "hgb_all": HGBVariantConfig(
        "hgb_all", use_vixlvl=True, use_oof=True, use_hyst=True, use_epw=True
    ),
}
VARIANT_ORDER = ["hgb_base", "hgb_vixlvl", "hgb_oof", "hgb_hyst", "hgb_epw", "hgb_all"]


# ── Mechanics ─────────────────────────────────────────────────────────────────


def apply_hysteresis(proba: np.ndarray, threshold: float, factor: float = 0.5) -> np.ndarray:
    """Hysteresis state machine on calibrated probabilities.

    Alert turns ON when proba >= threshold.
    Alert turns OFF when proba < factor * threshold.
    """
    alerts = np.zeros(len(proba), dtype=int)
    state = 0
    off_thr = threshold * factor
    for i, p in enumerate(proba):
        if state == 0:
            if p >= threshold:
                state = 1
        else:
            if p < off_thr:
                state = 0
        alerts[i] = state
    return alerts


def episode_sample_weights(
    y_train: np.ndarray,
    train_weeks: list[date],
    all_episodes: list[Episode],
) -> np.ndarray:
    """Sample weights: each episode in the training window has total weight 1.0.

    Calm weeks keep weight 1.0. Episode weeks get weight 1/N where N is
    the number of that episode's weeks present in the training window.
    """
    weights = np.ones(len(y_train), dtype=float)
    for ep in all_episodes:
        ep_idx = [i for i, w in enumerate(train_weeks) if ep.start <= w <= ep.end]
        if ep_idx:
            w_each = 1.0 / len(ep_idx)
            for idx in ep_idx:
                weights[idx] = w_each
    return weights


def _oof_scores_for_fold(
    X_all: np.ndarray,
    detect_arr: np.ndarray,
    all_weeks: list[date],
    core_ok: np.ndarray,
    fold: FoldSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute OOF decision_function scores over the pre-test history [0, test_lo).

    For each calendar year Y strictly before fold.test_year, trains a fresh HGB
    on all core-ok rows with week < start-of-Y, then scores the core-ok rows
    in year Y.

    Returns (oof_scores, oof_labels, oof_ku). oof_ku is all-False (no known-
    unlabeled tracking available for inner folds — conservative).

    Invariant: no row with index >= fold.test_lo is scored.
    """
    weeks_pretest = all_weeks[: fold.test_lo]
    unique_years = sorted({w.year for w in weeks_pretest})

    oof_scores: list[float] = []
    oof_labels: list[int] = []

    for yr in unique_years:
        yr_indices = [i for i, w in enumerate(weeks_pretest) if w.year == yr]
        yr_start = yr_indices[0]
        yr_end = yr_indices[-1] + 1  # exclusive

        # Inner train: all core-ok rows strictly before year yr
        inner_core = core_ok[:yr_start]
        X_inner = X_all[:yr_start][inner_core]
        y_inner = detect_arr[:yr_start][inner_core]

        if len(X_inner) < 20 or len(np.unique(y_inner)) < 2:
            continue

        # Inner val: core-ok rows in year yr
        inner_val_core = core_ok[yr_start:yr_end]
        X_inner_val = X_all[yr_start:yr_end][inner_val_core]
        y_inner_val = detect_arr[yr_start:yr_end][inner_val_core]

        if len(X_inner_val) == 0:
            continue

        pipe = make_hgb_pipeline()
        pipe.fit(X_inner, y_inner)
        scores = pipe.decision_function(X_inner_val)
        oof_scores.extend(scores.tolist())
        oof_labels.extend(y_inner_val.tolist())

    if not oof_scores:
        return (
            np.zeros(0, dtype=float),
            np.zeros(0, dtype=int),
            np.zeros(0, dtype=bool),
        )

    return (
        np.array(oof_scores, dtype=float),
        np.array(oof_labels, dtype=int),
        np.zeros(len(oof_scores), dtype=bool),
    )


# ── Core evaluation ───────────────────────────────────────────────────────────


def evaluate_hgb_variant(
    config: HGBVariantConfig,
    X_df: pd.DataFrame,
    vix_close: np.ndarray,
    labels_df: pd.DataFrame,
    folds: list[FoldSpec],
    ku_mask: np.ndarray,
    all_episodes: list[Episode],
) -> tuple[CandidateResult, list[ValFoldData]]:
    """Evaluate one HGB variant on all dev folds.

    Returns (CandidateResult, list[ValFoldData]) where ValFoldData stores
    calibrated val probabilities used for operating-curve re-thresholding.
    Hysteresis is post-processing; it is NOT exported to ONNX.
    """
    all_weeks: list[date] = list(labels_df["week"])
    detect_arr = labels_df["value"].values.astype(int)

    if config.use_vixlvl:
        vix_col = pd.Series(vix_close, index=X_df.index, name="vix_raw")
        X_eval = X_df.assign(vix_raw=vix_col)
    else:
        X_eval = X_df

    X_all = X_eval.values.astype(float) if hasattr(X_eval, "values") else np.asarray(X_eval)
    n_features = X_all.shape[1]
    core_ok = _core_feature_mask(X_eval, X_all)

    detect_folds: list[FoldResult] = []
    val_data_list: list[ValFoldData] = []
    last_pipe: Pipeline | None = None
    last_platt: PlattCalibrator | None = None

    for fold in folds:
        test_weeks_all = all_weeks[fold.test_lo : fold.test_hi]
        _guard_dev_only(test_weeks_all)

        core_train = core_ok[: fold.val_lo]
        core_val = core_ok[fold.val_lo : fold.test_lo]
        core_test = core_ok[fold.test_lo : fold.test_hi]

        X_train = X_all[: fold.val_lo][core_train]
        y_train = detect_arr[: fold.val_lo][core_train]
        train_weeks = [w for w, ok in zip(all_weeks[: fold.val_lo], core_train) if ok]

        X_val = X_all[fold.val_lo : fold.test_lo][core_val]
        y_val = detect_arr[fold.val_lo : fold.test_lo][core_val]
        val_ku = ku_mask[fold.val_lo : fold.test_lo][core_val]

        X_test = X_all[fold.test_lo : fold.test_hi][core_test]
        y_test = detect_arr[fold.test_lo : fold.test_hi][core_test]
        test_weeks = [w for w, ok in zip(test_weeks_all, core_test) if ok]
        rows_dropped = int((~core_test).sum())

        pipe = make_hgb_pipeline()
        if config.use_epw:
            weights = episode_sample_weights(y_train, train_weeks, all_episodes)
            pipe.fit(X_train, y_train, clf__sample_weight=weights)
        else:
            pipe.fit(X_train, y_train)

        if config.use_oof:
            oof_scores, oof_labels, _ = _oof_scores_for_fold(
                X_all, detect_arr, all_weeks, core_ok, fold
            )
            if len(oof_scores) >= 10 and len(np.unique(oof_labels)) >= 2:
                platt = PlattCalibrator(use_logit=False).fit(oof_scores, oof_labels)
            else:
                val_raw = pipe.decision_function(X_val)
                platt = PlattCalibrator(use_logit=False).fit(val_raw, y_val)
        else:
            val_raw = pipe.decision_function(X_val)
            platt = PlattCalibrator(use_logit=False).fit(val_raw, y_val)

        val_cal = platt.transform(pipe.decision_function(X_val))
        thr = choose_threshold(y_val, val_cal, FAR_BUDGET, excluded_mask=val_ku)

        val_data_list.append(
            ValFoldData(
                fold_year=fold.test_year,
                val_proba=val_cal,
                val_labels=y_val,
                val_ku=val_ku,
            )
        )

        test_raw = pipe.decision_function(X_test)
        test_cal = platt.transform(test_raw)

        if config.use_hyst:
            test_alerts = apply_hysteresis(test_cal, thr, factor=config.hyst_factor)
        else:
            test_alerts = (test_cal >= thr).astype(int)

        detect_folds.append(
            FoldResult(
                fold_year=fold.test_year,
                head="detect",
                weeks=test_weeks,
                true_labels=y_test,
                proba=test_cal,
                alerts=test_alerts,
                threshold=thr,
                platt_a=platt.a,  # type: ignore[arg-type]
                platt_b=platt.b,  # type: ignore[arg-type]
                rows_dropped=rows_dropped,
            )
        )
        last_pipe = pipe
        last_platt = platt

    # ONNX gate: hysteresis is serving-logic — only base probability model is exported.
    if detect_folds and last_pipe is not None:
        gate_rng = np.random.default_rng(42)
        X_gate = gate_rng.standard_normal((500, n_features)).astype(np.float32)
        X_gate[gate_rng.random((500, n_features)) < 0.15] = np.nan
        pipe_gate = copy.deepcopy(last_pipe)
        _set_njobs_1(pipe_gate)
        gate = check_onnx_gate(config.name, pipe_gate, last_platt, X_gate, n_features)
    else:
        gate = OnnxGateResult(passed=False, max_diff=None, error="No folds completed.")

    return (
        CandidateResult(
            name=config.name,
            detect_folds=detect_folds,
            predict_folds=[],
            onnx_gate=gate,
        ),
        val_data_list,
    )


# ── Operating curve ───────────────────────────────────────────────────────────


def operating_curve_hgb(
    detect_folds: list[FoldResult],
    val_data: list[ValFoldData],
    episodes: list[Episode],
    dev_weeks: list[date],
    budget: float,
    hyst_factor: float | None = None,
) -> tuple[int, float]:
    """Return (episodes_caught, fa_runs_per_year) at the given FAR budget.

    Re-thresholds each fold's calibrated val probabilities at `budget` to
    produce new test alerts.
    """
    all_alert_weeks: set[date] = set()
    for fr, vd in zip(detect_folds, val_data):
        thr = choose_threshold(vd.val_labels, vd.val_proba, budget, excluded_mask=vd.val_ku)
        if math.isinf(thr):
            continue
        if hyst_factor is not None:
            alerts = apply_hysteresis(fr.proba, thr, factor=hyst_factor)
        else:
            alerts = (fr.proba >= thr).astype(int)
        all_alert_weeks.update(w for w, a in zip(fr.weeks, alerts) if a == 1)

    ep_res = detect_event_results(episodes, all_alert_weeks)
    n_caught = sum(r.hit for r in ep_res)

    fa = false_alarm_runs(
        sorted(all_alert_weeks & set(dev_weeks)),
        dev_weeks,
        episodes,
        grace_weeks=GRACE_WEEKS,
        known_unlabeled=KNOWN_UNLABELED,
    )
    fa_real = [r for r in fa if not r.in_known_unlabeled]
    dev_years = sorted({w.year for w in dev_weeks})
    fa_per_year = len(fa_real) / max(len(dev_years), 1)
    return n_caught, fa_per_year


def _vix_level_operating_curve(
    folds: list[FoldSpec],
    labels_df: pd.DataFrame,
    vix_close: np.ndarray,
    ku_mask: np.ndarray,
    episodes: list[Episode],
    dev_weeks: list[date],
    budget: float,
) -> tuple[int, float]:
    """Return vix_level (episodes_caught, fa_runs_per_year) at the given FAR budget."""
    from .eval_harness import vix_rule

    all_weeks: list[date] = list(labels_df["week"])
    detect_arr = labels_df["value"].values.astype(int)

    all_alert_weeks: set[date] = set()
    for fold in folds:
        val_detect = detect_arr[fold.val_lo : fold.test_lo]
        val_ku = ku_mask[fold.val_lo : fold.test_lo]
        val_vix = vix_close[fold.val_lo : fold.test_lo]
        thr = choose_threshold(val_detect, val_vix, budget, excluded_mask=val_ku)
        if math.isinf(thr):
            continue
        test_vix = vix_close[fold.test_lo : fold.test_hi]
        test_weeks_fold = all_weeks[fold.test_lo : fold.test_hi]
        alerts = vix_rule(test_vix, thr)
        all_alert_weeks.update(w for w, a in zip(test_weeks_fold, alerts) if a == 1)

    ep_res = detect_event_results(episodes, all_alert_weeks)
    n_caught = sum(r.hit for r in ep_res)

    fa = false_alarm_runs(
        sorted(all_alert_weeks & set(dev_weeks)),
        dev_weeks,
        episodes,
        grace_weeks=GRACE_WEEKS,
        known_unlabeled=KNOWN_UNLABELED,
    )
    fa_real = [r for r in fa if not r.in_known_unlabeled]
    dev_years = sorted({w.year for w in dev_weeks})
    fa_per_year = len(fa_real) / max(len(dev_years), 1)
    return n_caught, fa_per_year


# ── Report ────────────────────────────────────────────────────────────────────


def write_hgb_report(
    out_path: Path,
    results: dict[str, CandidateResult],
    pooled: dict[str, dict],
    boot: dict[str, dict],
    op_curves: dict[str, list[tuple[int, float]]],
    episodes: list[Episode],
    label_set: str,
    first_test_year: int,
    last_test_year: int,
) -> None:
    today = date.today().isoformat()
    all_names = VARIANT_ORDER + ["vix_level"]
    n_dev_years = last_test_year - first_test_year + 1

    ep_years: set[int] = set()
    for ep in episodes:
        d = ep.start
        while d <= ep.end:
            ep_years.add(d.year)
            d += timedelta(weeks=1)
    calm_years = [y for y in range(first_test_year, last_test_year + 1) if y not in ep_years]

    def _ci_str(ci: list[float] | None, decimals: int = 3) -> str:
        if ci is None:
            return "—"
        fmt = f".{decimals}f"
        return f"[{ci[0]:{fmt}}, {ci[1]:{fmt}}]"

    lines: list[str] = [
        f"# HGB Experiment Report — {label_set} ({first_test_year}–{last_test_year})",
        "",
        f"Generated: {today}  ",
        f"Label set: `{label_set}` · FAR budget = {FAR_BUDGET:.0%}  ",
        f"Dev folds: test years {first_test_year}–{last_test_year}  ",
        "",
        "> **Pre-registered**: variants specified before seeing results.",
        "> No post-hoc tuning. VIX level is the evaluation bar.",
        "> Hysteresis is serving-logic — not part of the ONNX graph.",
        "",
        "---",
        "",
        "## Summary table",
        "",
        "| Variant | Event F1 | Ep hits | Row F1 | PR AUC | Brier | FA calm | FA total | ONNX |",
        "|---------|----------|---------|--------|--------|-------|---------|----------|------|",
    ]

    for nm in all_names:
        if nm not in pooled:
            lines.append(f"| {nm} | — | — | — | — | — | — | — | — |")
            continue
        m = pooled[nm]
        cr = results.get(nm)
        gate_obj = cr.onnx_gate if cr else None
        if gate_obj is None:
            gate_str = "n/a"
        elif gate_obj.passed:
            gate_str = "pass"
        else:
            gate_str = f"FAIL: {gate_obj.error}"
        pm = m.get("prob_metrics") or {}
        pr_auc = f"{pm.get('pr_auc', float('nan')):.3f}" if pm else "—"
        brier = f"{pm.get('brier', float('nan')):.3f}" if pm else "—"
        hits = f"{m['episode_hits']}/{m['episodes_total']}"
        lines.append(
            f"| {nm} | {m['event_f1']:.3f} | {hits} | {m['row']['f1']:.3f}"
            f" | {pr_auc} | {brier}"
            f" | {m.get('fa_calm_runs', '—')} | {m.get('fa_total_runs', '—')}"
            f" | {gate_str} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Bootstrap CIs (year-block, 2000 resamples)",
        "",
        "| Variant | Event F1 | 95% CI | Row F1 CI | PR AUC CI | Brier CI |",
        "|---------|----------|--------|-----------|-----------|---------|",
    ]
    for nm in all_names:
        if nm not in boot or nm not in pooled:
            lines.append(f"| {nm} | — | — | — | — | — |")
            continue
        b = boot[nm]
        ef1 = pooled[nm]["event_f1"]
        lines.append(
            f"| {nm} | {ef1:.3f} | {_ci_str(b.get('event_f1_ci95'))}"
            f" | {_ci_str(b.get('row_f1_ci95'))}"
            f" | {_ci_str(b.get('pr_auc_ci95'))}"
            f" | {_ci_str(b.get('brier_ci95'))} |"
        )

    lines += [
        "",
        "## Paired bootstrap differences (variant − vix_level)",
        "",
        "Positive = variant better than vix_level.",
        "",
        "| Variant | ΔEvent F1 | 95% CI | ΔRow F1 | 95% CI |",
        "|---------|-----------|--------|---------|--------|",
    ]
    vix_boot = boot.get("vix_level", {})
    for nm in VARIANT_ORDER:
        if nm not in boot:
            lines.append(f"| {nm} | — | — | — | — |")
            continue
        b_ef1_a = boot[nm].get("_boot_event_f1", [])
        b_ef1_v = vix_boot.get("_boot_event_f1", [])
        b_rf1_a = boot[nm].get("_boot_row_f1", [])
        b_rf1_v = vix_boot.get("_boot_row_f1", [])
        n = min(len(b_ef1_a), len(b_ef1_v))
        if n > 0:
            diff_ef1 = np.array(b_ef1_a[:n]) - np.array(b_ef1_v[:n])
            diff_rf1 = np.array(b_rf1_a[:n]) - np.array(b_rf1_v[:n])
            ef1_ci = [float(np.percentile(diff_ef1, 2.5)), float(np.percentile(diff_ef1, 97.5))]
            rf1_ci = [float(np.percentile(diff_rf1, 2.5)), float(np.percentile(diff_rf1, 97.5))]
            lines.append(
                f"| {nm} | {float(np.mean(diff_ef1)):+.3f} | {_ci_str(ef1_ci)}"
                f" | {float(np.mean(diff_rf1)):+.3f} | {_ci_str(rf1_ci)} |"
            )
        else:
            lines.append(f"| {nm} | — | — | — | — |")

    lines += [
        "",
        "## Paired bootstrap differences (variant − hgb_base)",
        "",
        "| Variant | ΔEvent F1 | 95% CI | ΔRow F1 | 95% CI |",
        "|---------|-----------|--------|---------|--------|",
    ]
    base_boot = boot.get("hgb_base", {})
    for nm in VARIANT_ORDER[1:]:
        if nm not in boot:
            lines.append(f"| {nm} | — | — | — | — |")
            continue
        b_ef1_a = boot[nm].get("_boot_event_f1", [])
        b_ef1_b = base_boot.get("_boot_event_f1", [])
        b_rf1_a = boot[nm].get("_boot_row_f1", [])
        b_rf1_b = base_boot.get("_boot_row_f1", [])
        n = min(len(b_ef1_a), len(b_ef1_b))
        if n > 0:
            diff_ef1 = np.array(b_ef1_a[:n]) - np.array(b_ef1_b[:n])
            diff_rf1 = np.array(b_rf1_a[:n]) - np.array(b_rf1_b[:n])
            ef1_ci = [float(np.percentile(diff_ef1, 2.5)), float(np.percentile(diff_ef1, 97.5))]
            rf1_ci = [float(np.percentile(diff_rf1, 2.5)), float(np.percentile(diff_rf1, 97.5))]
            lines.append(
                f"| {nm} | {float(np.mean(diff_ef1)):+.3f} | {_ci_str(ef1_ci)}"
                f" | {float(np.mean(diff_rf1)):+.3f} | {_ci_str(rf1_ci)} |"
            )
        else:
            lines.append(f"| {nm} | — | — | — | — |")

    lines += [
        "",
        "---",
        "",
        "## Per-episode hit table",
        "",
        "| # | Start | End | Wks | " + " | ".join(all_names) + " |",
        "|---|-------|-----|-----|" + "---|" * len(all_names),
    ]
    for i, ep in enumerate(episodes, 1):
        row_parts = [str(i), str(ep.start), str(ep.end), str(ep.length)]
        for nm in all_names:
            if nm not in pooled:
                row_parts.append("—")
                continue
            ep_results = pooled[nm].get("ep_results", [])
            hit = ep_results[i - 1].hit if i - 1 < len(ep_results) else False
            row_parts.append("Y" if hit else "N")
        lines.append("| " + " | ".join(row_parts) + " |")

    lines += [
        "",
        "---",
        "",
        "## Operating curve at FAR budgets 2%, 5%, 10%",
        "",
        f"FA/yr denominator: {n_dev_years} dev years ({first_test_year}–{last_test_year}).  ",
        f"Episodes total: {len(episodes)}.  ",
        f"Calm years (no episode): {calm_years}",
        "",
        "| Variant | Budget | Ep caught | FA/yr |",
        "|---------|--------|-----------|-------|",
    ]
    for nm in all_names:
        if nm not in op_curves:
            for b in OPERATING_CURVE_BUDGETS:
                lines.append(f"| {nm} | {b:.0%} | — | — |")
            continue
        for (n_caught, fa_yr), b in zip(op_curves[nm], OPERATING_CURVE_BUDGETS):
            lines.append(f"| {nm} | {b:.0%} | {n_caught}/{len(episodes)} | {fa_yr:.2f} |")

    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ── CLI entries ───────────────────────────────────────────────────────────────


def run_hgb_experiments(
    session,
    out_dir: Path,
    label_set: str = "ews_v1",
    first_test_year: int = DEV_FIRST_YEAR,
    last_test_year: int = DEV_LAST_YEAR,
    n_boot: int = N_BOOT,
    skip_bootstrap: bool = False,
) -> int:
    """Load data, run all six HGB variants, write report."""
    from .episodes import extract_episodes
    from .eval_harness import make_folds_v2
    from .features.build import build_from_db, weekly_close
    from .features.spec import SPEC_V1
    from .labels import load_labels
    from .selection import _evaluate_vix_baselines

    labels_df = load_labels(session, label_set=label_set)
    if labels_df.empty:
        print(f"ERROR: no labels for {label_set!r}")
        return 1

    weeks: list[date] = list(labels_df["week"])
    feature_df = build_from_db(session, SPEC_V1, weeks, as_of=None)
    vix_close = weekly_close(session, "vix", weeks)
    if np.all(np.isnan(vix_close)):
        print("ERROR: VIX weekly close is entirely NaN.")
        return 1

    ku_mask = np.zeros(len(weeks), dtype=bool)
    for ks, ke, _ in KNOWN_UNLABELED:
        for i, w in enumerate(weeks):
            if ks <= w <= ke:
                ku_mask[i] = True

    folds = make_folds_v2(
        weeks,
        labels_df["value"].values.astype(int),
        first_test_year=first_test_year,
        last_test_year=last_test_year,
    )

    ep2 = extract_episodes(labels_df, merge_gap=2)
    dev_start = date(first_test_year, 1, 1)
    dev_end = date(last_test_year, 12, 31)
    episodes = [e for e in ep2 if dev_start <= e.start <= dev_end]
    dev_weeks = [w for w in weeks if dev_start <= w <= dev_end]

    ep_years: set[int] = set()
    for ep in episodes:
        d = ep.start
        while d <= ep.end:
            ep_years.add(d.year)
            d += timedelta(weeks=1)
    calm_years = [y for y in range(first_test_year, last_test_year + 1) if y not in ep_years]

    results: dict[str, CandidateResult] = {}
    val_data_map: dict[str, list[ValFoldData]] = {}

    for vname in VARIANT_ORDER:
        config = VARIANTS[vname]
        print(f"hgb-experiments: evaluating {vname} ...")
        cr, val_data = evaluate_hgb_variant(
            config, feature_df, vix_close, labels_df, folds, ku_mask, ep2
        )
        results[vname] = cr
        val_data_map[vname] = val_data

    vix_baselines = _evaluate_vix_baselines(labels_df, folds, vix_close, ku_mask)
    vix_result = vix_baselines["vix_level"]
    results["vix_level"] = vix_result

    all_names_with_vix = VARIANT_ORDER + ["vix_level"]
    pooled: dict[str, dict] = {}
    for nm in all_names_with_vix:
        cr = results.get(nm)
        if cr and cr.detect_folds:
            pooled[nm] = compute_pooled_metrics(cr.detect_folds, episodes, dev_weeks, calm_years)

    boot: dict[str, dict] = {}
    if not skip_bootstrap:
        for nm in all_names_with_vix:
            cr = results.get(nm)
            if cr and cr.detect_folds:
                print(f"hgb-experiments: bootstrapping {nm} ...")
                boot[nm] = year_block_bootstrap(
                    cr.detect_folds, episodes, dev_weeks, n_boot=n_boot, seed=0
                )
    else:
        for nm in all_names_with_vix:
            boot[nm] = {
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

    op_curves: dict[str, list[tuple[int, float]]] = {}
    for vname in VARIANT_ORDER:
        config = VARIANTS[vname]
        cr = results.get(vname)
        vd = val_data_map.get(vname, [])
        if cr and cr.detect_folds and vd:
            hyst_factor = config.hyst_factor if config.use_hyst else None
            op_curves[vname] = [
                operating_curve_hgb(
                    cr.detect_folds, vd, episodes, dev_weeks, b, hyst_factor=hyst_factor
                )
                for b in OPERATING_CURVE_BUDGETS
            ]
    op_curves["vix_level"] = [
        _vix_level_operating_curve(folds, labels_df, vix_close, ku_mask, episodes, dev_weeks, b)
        for b in OPERATING_CURVE_BUDGETS
    ]

    today = date.today().isoformat()
    out_path = out_dir / f"{today}-hgb-experiments.md"
    write_hgb_report(
        out_path,
        results,
        pooled,
        boot,
        op_curves,
        episodes,
        label_set,
        first_test_year,
        last_test_year,
    )
    print(f"hgb-experiments: report → {out_path}")
    return 0


def run_export_hgb_variant(
    session,
    out_path: Path,
    variant_name: str,
    label_set: str = "ews_v1",
    first_test_year: int = DEV_FIRST_YEAR,
    last_test_year: int = DEV_LAST_YEAR,
) -> int:
    """Rerun one HGB variant on dev folds; write one row per dev week.

    Columns: week, fold_year, label, episode_id, known_unlabeled, vix_close,
    <variant>_prob, <variant>_threshold, <variant>_alert.
    Thresholds are the per-fold values actually used.
    """
    if variant_name not in VARIANTS:
        print(f"ERROR: unknown variant {variant_name!r}. Known: {list(VARIANTS)}")
        return 1

    from .episodes import extract_episodes
    from .eval_harness import make_folds_v2
    from .features.build import build_from_db, weekly_close
    from .features.spec import SPEC_V1
    from .labels import load_labels

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

    folds = make_folds_v2(
        weeks,
        labels_df["value"].values.astype(int),
        first_test_year=first_test_year,
        last_test_year=last_test_year,
    )

    ep2 = extract_episodes(labels_df, merge_gap=2)
    dev_start = date(first_test_year, 1, 1)
    dev_end = date(last_test_year, 12, 31)
    episodes = [e for e in ep2 if dev_start <= e.start <= dev_end]

    config = VARIANTS[variant_name]
    print(f"export-dev-predictions: evaluating {variant_name} ...")
    cr, _ = evaluate_hgb_variant(config, feature_df, vix_close, labels_df, folds, ku_mask, ep2)

    week_to_vix = {w: float(vix_close[i]) for i, w in enumerate(weeks)}
    week_to_ku = {w: bool(ku_mask[i]) for i, w in enumerate(weeks)}
    detect_by_week = {w: int(v) for w, v in zip(weeks, labels_df["value"].values)}

    week_to_ep: dict[date, int] = {}
    for ep_idx, ep in enumerate(episodes, 1):
        d = ep.start
        while d <= ep.end:
            week_to_ep[d] = ep_idx
            d += timedelta(weeks=1)

    rows: list[dict] = []
    for fr in cr.detect_folds:
        for w, prob, alert in zip(fr.weeks, fr.proba, fr.alerts):
            rows.append(
                {
                    "week": w,
                    "fold_year": fr.fold_year,
                    "label": detect_by_week.get(w, None),
                    "episode_id": week_to_ep.get(w, None),
                    "known_unlabeled": int(week_to_ku.get(w, False)),
                    "vix_close": week_to_vix.get(w, float("nan")),
                    f"{variant_name}_prob": float(prob),
                    f"{variant_name}_threshold": float(fr.threshold),
                    f"{variant_name}_alert": int(alert),
                }
            )

    col_order = [
        "week",
        "fold_year",
        "label",
        "episode_id",
        "known_unlabeled",
        "vix_close",
        f"{variant_name}_prob",
        f"{variant_name}_threshold",
        f"{variant_name}_alert",
    ]
    df = pd.DataFrame(sorted(rows, key=lambda r: r["week"]))[col_order]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    n_alerts = int((df[f"{variant_name}_alert"] == 1).sum())
    print(
        f"export-dev-predictions: {len(df)} dev weeks"
        f" | {variant_name} alerts={n_alerts} → {out_path}"
    )
    return 0
