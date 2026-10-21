"""Phase 2 Step 2.4 — candidate model selection on development folds.

Candidates: LogReg, HGB, Two-stage (IsoForest + StackingClassifier).
Each candidate is evaluated with walk-forward folds (test years 2008-2017 only).
Platt calibration (explicit a, b) is fitted on val scores.
Threshold is chosen on calibrated val probabilities at 5% calm-week FAR.
ONNX gate: candidates failing export or parity check are ineligible for selection.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .calibration import PlattCalibrator
from .episodes import Episode
from .eval_harness import (
    KNOWN_UNLABELED,
    FoldSpec,
    choose_threshold,
    detect_event_results,
    false_alarm_runs,
    predict_event_results,
    row_metrics,
)

DEV_FIRST_YEAR = 2008
DEV_LAST_YEAR = 2017
DEV_CUTOFF = date(2018, 1, 1)
GRACE_WEEKS = 2
FAR_BUDGET = 0.05
N_BOOT = 2000
HORIZON = 4


# ── Dataclasses ───────────────────────────────────────────────────────────────


@dataclass
class FoldResult:
    fold_year: int
    head: str  # "detect" or "predict"
    weeks: list[date]
    true_labels: np.ndarray
    proba: np.ndarray
    alerts: np.ndarray
    threshold: float
    platt_a: float
    platt_b: float
    score_status: str = "ok"  # "ok" | "insufficient_data" | "single_class_val"
    rows_dropped: int = 0


@dataclass
class OnnxGateResult:
    passed: bool
    max_diff: float | None
    error: str | None


@dataclass
class CandidateResult:
    name: str
    detect_folds: list[FoldResult]
    predict_folds: list[FoldResult]
    onnx_gate: OnnxGateResult


# ── Custom transformers ───────────────────────────────────────────────────────


class MedianImputeWithIndicator(BaseEstimator, TransformerMixin):
    """Median-impute, then append one 0/1 'was missing' column per feature.

    Medians are computed from fit rows only; transform rows use the stored medians.
    This is ONNX-exportable via the converter registered in calibration.py.
    """

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        med = np.nanmedian(X, axis=0)
        self.medians_ = np.where(np.isnan(med), 0.0, med)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        nan = np.isnan(X)
        return np.hstack([np.where(nan, self.medians_, X), nan.astype(float)])


class IsoForestAppender(BaseEstimator, TransformerMixin):
    """Fits IsolationForest; appends its decision_function score as a new column."""

    def __init__(
        self,
        n_estimators: int = 500,
        contamination: str | float = "auto",
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> IsoForestAppender:
        self.iso_ = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self.iso_.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        scores = self.iso_.decision_function(np.asarray(X)).reshape(-1, 1)
        return np.hstack([np.asarray(X), scores])

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            names = list(input_features)
        else:
            names = [f"x{i}" for i in range(self.iso_.n_features_in_)]
        return names + ["iso_score"]


# ── Model factories ───────────────────────────────────────────────────────────


def _core_feature_mask(X_df, X_arr: np.ndarray) -> np.ndarray:
    """Return bool array: True = row has all core features present.

    Core instruments: vix, msci_usa, us_10y_yield.  Core columns are those
    whose FeatureSpec.instrument_id is in that set (e.g. sp500_lr_1w comes
    from instrument_id="msci_usa"; spread_10y_3m from "us_10y_yield").
    A row is excluded if NaN in ANY core column.
    If no core columns exist (e.g. synthetic test data), all rows are kept.
    """
    if not hasattr(X_df, "columns"):
        return np.ones(len(X_arr), dtype=bool)

    from .features.spec import SPEC_V1

    core_instruments = {"vix", "msci_usa", "us_10y_yield"}
    core_cols = {s.col for s in SPEC_V1 if s.instrument_id in core_instruments}
    df_cols = list(X_df.columns)
    core_idx = [i for i, c in enumerate(df_cols) if str(c) in core_cols]
    if not core_idx:
        return np.ones(len(X_arr), dtype=bool)
    core_sub = X_arr[:, core_idx]
    return ~np.any(np.isnan(core_sub), axis=1)


def make_logreg_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", MedianImputeWithIndicator()),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def make_hgb_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_depth=3,
                    learning_rate=0.05,
                    max_iter=200,
                    l2_regularization=1.0,
                    random_state=42,
                ),
            ),
        ]
    )


def make_twostage_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", MedianImputeWithIndicator()),
            (
                "iso_appender",
                IsoForestAppender(n_estimators=500, contamination="auto", random_state=42),
            ),
            (
                "clf",
                StackingClassifier(
                    estimators=[
                        (
                            "dt",
                            DecisionTreeClassifier(
                                max_depth=10, class_weight="balanced", random_state=42
                            ),
                        ),
                        (
                            "rf",
                            RandomForestClassifier(
                                n_estimators=200,
                                class_weight="balanced",
                                random_state=42,
                                n_jobs=-1,
                            ),
                        ),
                        (
                            "lr",
                            LogisticRegression(
                                C=0.1, class_weight="balanced", solver="liblinear", max_iter=1000
                            ),
                        ),
                    ],
                    final_estimator=LogisticRegression(C=1.0, solver="liblinear", max_iter=1000),
                    cv=5,
                    passthrough=False,
                    n_jobs=-1,
                ),
            ),
        ]
    )


CANDIDATES: dict[str, Callable[[], Pipeline]] = {
    "logreg": make_logreg_pipeline,
    "hgb": make_hgb_pipeline,
    "twostage": make_twostage_pipeline,
}
SIMPLICITY_ORDER = ["logreg", "hgb", "twostage"]

# logreg and hgb expose decision_function (raw log-odds); twostage uses predict_proba.
# PlattCalibrator(use_logit=False) fits on the raw margin directly — no saturation risk.
_CANDIDATE_USES_LOGIT: dict[str, bool] = {
    "logreg": False,
    "hgb": False,
    "twostage": True,
}


# ── ONNX helpers ─────────────────────────────────────────────────────────────


def _set_njobs_1(pipeline: Pipeline) -> None:
    """Recursively set n_jobs=1 on all sub-estimators (for ONNX determinism).

    Only called on a deepcopy used for the gate; training pipeline stays parallel.
    """
    for _, step in pipeline.steps:
        if hasattr(step, "n_jobs"):
            step.n_jobs = 1
        if hasattr(step, "estimators"):
            for _, est in step.estimators:
                if hasattr(est, "n_jobs"):
                    est.n_jobs = 1
        if hasattr(step, "final_estimator") and hasattr(step.final_estimator, "n_jobs"):
            step.final_estimator.n_jobs = 1


# ── Guard ─────────────────────────────────────────────────────────────────────


def _guard_dev_only(weeks: list[date]) -> None:
    """Raise if any week is on or after DEV_CUTOFF (2018-01-01)."""
    bad = [w for w in weeks if w >= DEV_CUTOFF]
    if bad:
        raise ValueError(
            f"Dev evaluation must not touch weeks >= {DEV_CUTOFF}. "
            f"Found {len(bad)} such weeks: {bad[:3]}..."
        )


# ── Event metrics ─────────────────────────────────────────────────────────────


def _count_alert_runs(alert_set: set[date], all_weeks: list[date]) -> int:
    count = 0
    in_run = False
    for w in sorted(all_weeks):
        if w in alert_set:
            if not in_run:
                count += 1
                in_run = True
        else:
            in_run = False
    return count


def event_precision_recall_f1(
    episodes: list[Episode],
    alert_set: set[date],
    all_weeks: list[date],
    grace_weeks: int = GRACE_WEEKS,
    known_unlabeled: list | None = None,
) -> tuple[float, float, float]:
    """Compute event-level precision, recall, F1.

    event_recall = episodes with >=1 alert / total episodes
    event_precision = alert runs overlapping an episode +-grace_weeks / all alert runs
    Known-unlabeled windows are excluded from false-alarm counts (consistent with
    choose_threshold which also excludes them).
    """
    if known_unlabeled is None:
        known_unlabeled = KNOWN_UNLABELED

    if not episodes:
        return 0.0, 0.0, 0.0

    ep_results = detect_event_results(episodes, alert_set)
    recall = sum(r.hit for r in ep_results) / len(episodes)

    total_runs = _count_alert_runs(alert_set, all_weeks)
    if total_runs == 0:
        return 0.0, recall, 0.0

    fa_all = false_alarm_runs(
        sorted(alert_set & set(all_weeks)),
        all_weeks,
        episodes,
        grace_weeks=grace_weeks,
        known_unlabeled=known_unlabeled,
    )
    # Exclude runs inside known-unlabeled windows from both FA count and total
    fa_real = [r for r in fa_all if not r.in_known_unlabeled]
    # Runs in KU windows are also excluded from the denominator
    ku_runs = len(fa_all) - len(fa_real)
    effective_total = total_runs - ku_runs
    if effective_total <= 0:
        precision = 1.0
    else:
        precision = max(0.0, (effective_total - len(fa_real)) / effective_total)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


# ── Bootstrap ─────────────────────────────────────────────────────────────────


def year_block_bootstrap(
    detect_folds: list[FoldResult],
    episodes: list[Episode],
    all_dev_weeks: list[date],
    n_boot: int = N_BOOT,
    seed: int = 0,
) -> dict:
    """Resample test years; compute event F1, row F1, PR AUC, Brier, log loss."""
    from sklearn.metrics import average_precision_score, brier_score_loss
    from sklearn.metrics import log_loss as sklearn_log_loss

    rng = np.random.default_rng(seed)
    years = sorted({fr.fold_year for fr in detect_folds})
    results_by_year = {fr.fold_year: fr for fr in detect_folds}

    eps_by_year: dict[int, list[Episode]] = {}
    for ep in episodes:
        eps_by_year.setdefault(ep.start.year, []).append(ep)

    boot_event_f1: list[float] = []
    boot_row_f1: list[float] = []
    boot_pr_auc: list[float] = []
    boot_brier: list[float] = []
    boot_ll: list[float] = []

    for _ in range(n_boot):
        sampled_years = rng.choice(years, size=len(years), replace=True)

        y_true_list: list[np.ndarray] = []
        y_proba_list: list[np.ndarray] = []
        y_alert_list: list[np.ndarray] = []
        pool_alert: set[date] = set()
        pool_weeks: list[date] = []
        pool_eps: list[Episode] = []

        for yr in sampled_years:
            fr = results_by_year[int(yr)]
            y_true_list.append(fr.true_labels)
            y_proba_list.append(fr.proba)
            y_alert_list.append(fr.alerts)
            for w, a in zip(fr.weeks, fr.alerts):
                pool_weeks.append(w)
                if a == 1:
                    pool_alert.add(w)
            pool_eps.extend(eps_by_year.get(int(yr), []))

        y_true = np.concatenate(y_true_list)
        y_proba = np.concatenate(y_proba_list)
        y_alert = np.concatenate(y_alert_list)

        m = row_metrics(y_true, y_alert)
        boot_row_f1.append(m["f1"])

        _, _, ef1 = event_precision_recall_f1(pool_eps, pool_alert, sorted(set(pool_weeks)))
        boot_event_f1.append(ef1)

        if len(np.unique(y_true)) >= 2:
            boot_pr_auc.append(float(average_precision_score(y_true, y_proba)))
            boot_brier.append(float(brier_score_loss(y_true, y_proba)))
            boot_ll.append(float(sklearn_log_loss(y_true, y_proba)))
        else:
            boot_pr_auc.append(float("nan"))
            boot_brier.append(float("nan"))
            boot_ll.append(float("nan"))

    def _ci(vals: list[float]) -> list[float] | None:
        v = np.asarray([x for x in vals if not math.isnan(x)])
        if len(v) == 0:
            return None
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    def _se(vals: list[float]) -> float:
        v = np.asarray([x for x in vals if not math.isnan(x)])
        return float(np.std(v)) if len(v) > 0 else float("nan")

    return {
        "n_boot": n_boot,
        "event_f1_ci95": _ci(boot_event_f1),
        "event_f1_se": _se(boot_event_f1),
        "row_f1_ci95": _ci(boot_row_f1),
        "pr_auc_ci95": _ci(boot_pr_auc),
        "brier_ci95": _ci(boot_brier),
        "log_loss_ci95": _ci(boot_ll),
        "_boot_event_f1": boot_event_f1,
        "_boot_row_f1": boot_row_f1,
    }


# ── ONNX gate ─────────────────────────────────────────────────────────────────


def check_onnx_gate(
    name: str,
    pipeline: Pipeline,
    platt: PlattCalibrator,
    X_sample: np.ndarray,
    n_features: int,
) -> OnnxGateResult:
    """Test ONNX export + parity with sklearn. Returns OnnxGateResult."""
    try:
        import onnxruntime as ort

        from .calibration import onnx_export_with_platt
    except ImportError as e:
        return OnnxGateResult(passed=False, max_diff=None, error=f"ImportError: {e}")

    try:
        onnx_bytes = onnx_export_with_platt(pipeline, platt, n_features, X_sample=X_sample[:5])
    except Exception as e:
        return OnnxGateResult(passed=False, max_diff=None, error=f"Export failed: {e}")

    try:
        sess = ort.InferenceSession(onnx_bytes)
        input_name = sess.get_inputs()[0].name
        X_f32 = X_sample.astype(np.float32)
        ort_probs = sess.run(["calibrated_probability"], {input_name: X_f32})[0].flatten()

        if platt.use_logit:
            sklearn_raw = pipeline.predict_proba(X_sample)[:, 1]
        else:
            sklearn_raw = pipeline.decision_function(X_sample)
        sklearn_probs = platt.transform(sklearn_raw)

        max_diff = float(np.max(np.abs(ort_probs - sklearn_probs.astype(np.float32))))
        passed = max_diff <= 1e-5
        return OnnxGateResult(passed=passed, max_diff=max_diff, error=None)
    except Exception as e:
        return OnnxGateResult(passed=False, max_diff=None, error=f"Runtime error: {e}")


# ── Selection rule ────────────────────────────────────────────────────────────


def apply_selection_rule(
    candidate_metrics: dict[str, dict],
    candidate_boot: dict[str, dict],
    onnx_gates: dict[str, OnnxGateResult],
) -> tuple[str, str]:
    """Return (selected_name, reason_string).

    Rule: best pooled dev event F1 → fewer FA runs per calm year → log loss.
    Simplest model within one bootstrap SE of the best wins.
    Candidates failing the ONNX gate are excluded.
    """
    eligible = {n for n, g in onnx_gates.items() if g.passed}
    if not eligible:
        return "", "No candidate passed the ONNX gate."

    names = [n for n in SIMPLICITY_ORDER if n in candidate_metrics and n in eligible]
    if not names:
        return "", "No ONNX-eligible candidates."

    def sort_key(n: str) -> tuple:
        m = candidate_metrics[n]
        return (-m["event_f1"], m.get("fa_per_calm", 0.0), m.get("log_loss", 0.0))

    sorted_names = sorted(names, key=sort_key)
    best = sorted_names[0]
    best_ef1 = candidate_metrics[best]["event_f1"]
    best_se = candidate_boot[best]["event_f1_se"]

    within_se = [n for n in names if candidate_metrics[n]["event_f1"] >= best_ef1 - best_se]

    for n in SIMPLICITY_ORDER:
        if n in within_se:
            if n != best:
                reason = (
                    f"Best pooled event F1 was {best} ({best_ef1:.3f}). "
                    f"{n} is within 1 bootstrap SE ({best_se:.3f}) and simpler."
                )
            else:
                reason = f"Best pooled event F1: {best_ef1:.3f}; also simplest within 1 SE."
            return n, reason

    return best, f"Best pooled event F1: {best_ef1:.3f}."


# ── Core evaluation ───────────────────────────────────────────────────────────


def evaluate_candidate(
    name: str,
    make_pipeline_fn: Callable[[], Pipeline],
    X_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    folds: list[FoldSpec],
    ku_mask: np.ndarray,
    X_sample_for_gate: np.ndarray | None = None,
) -> CandidateResult:
    """Evaluate one candidate on all dev folds.

    Fit PlattCalibrator on val scores. Choose threshold on calibrated val probs
    at FAR_BUDGET. No test data touches the train/val pipeline.
    Rows where vix_lr_1w is NaN are excluded from fit, val, and test scoring.
    """
    from .labels import build_targets

    all_weeks: list[date] = list(labels_df["week"])
    detect_arr = labels_df["value"].values.astype(int)
    targets = build_targets(labels_df, horizon=HORIZON)
    predict_arr = targets["predict"].values

    X_all = X_df.values.astype(float) if hasattr(X_df, "values") else np.asarray(X_df)
    n_features = X_all.shape[1]

    # Core-feature mask: rows with all core features present (vix_*, msci_usa_*, us_10y_yield_*)
    core_ok = _core_feature_mask(X_df, X_all)

    detect_folds: list[FoldResult] = []
    predict_folds: list[FoldResult] = []

    use_logit = _CANDIDATE_USES_LOGIT.get(name, True)

    last_pipe_d: Pipeline | None = None
    last_platt_d: PlattCalibrator | None = None

    for fold in folds:
        test_weeks_all = all_weeks[fold.test_lo : fold.test_hi]
        _guard_dev_only(test_weeks_all)

        X_train_raw = X_all[: fold.val_lo]
        X_val_raw = X_all[fold.val_lo : fold.test_lo]
        X_test_raw = X_all[fold.test_lo : fold.test_hi]

        y_train_det_raw = detect_arr[: fold.val_lo]
        y_val_det_raw = detect_arr[fold.val_lo : fold.test_lo]
        y_test_det_raw = detect_arr[fold.test_lo : fold.test_hi]
        val_ku_raw = ku_mask[fold.val_lo : fold.test_lo]

        core_train = core_ok[: fold.val_lo]
        core_val = core_ok[fold.val_lo : fold.test_lo]
        core_test = core_ok[fold.test_lo : fold.test_hi]

        X_train = X_train_raw[core_train]
        y_train_det = y_train_det_raw[core_train]
        X_val = X_val_raw[core_val]
        y_val_det = y_val_det_raw[core_val]
        val_ku = val_ku_raw[core_val]
        X_test = X_test_raw[core_test]
        y_test_det = y_test_det_raw[core_test]
        test_weeks = [w for w, ok in zip(test_weeks_all, core_test) if ok]
        rows_dropped = int((~core_test).sum())

        # detect head
        pipe_d = make_pipeline_fn()
        pipe_d.fit(X_train, y_train_det)

        if use_logit:
            val_raw_d = pipe_d.predict_proba(X_val)[:, 1]
        else:
            val_raw_d = pipe_d.decision_function(X_val)
        platt_d = PlattCalibrator(use_logit=use_logit).fit(val_raw_d, y_val_det)
        val_cal_d = platt_d.transform(val_raw_d)
        thr_d = choose_threshold(y_val_det, val_cal_d, FAR_BUDGET, excluded_mask=val_ku)

        if use_logit:
            test_raw_d = pipe_d.predict_proba(X_test)[:, 1]
        else:
            test_raw_d = pipe_d.decision_function(X_test)
        test_cal_d = platt_d.transform(test_raw_d)
        test_alerts_d = (test_cal_d >= thr_d).astype(int)

        detect_folds.append(
            FoldResult(
                fold_year=fold.test_year,
                head="detect",
                weeks=test_weeks,
                true_labels=y_test_det,
                proba=test_cal_d,
                alerts=test_alerts_d,
                threshold=thr_d,
                platt_a=platt_d.a,  # type: ignore[arg-type]
                platt_b=platt_d.b,  # type: ignore[arg-type]
                rows_dropped=rows_dropped,
            )
        )
        last_pipe_d = pipe_d
        last_platt_d = platt_d

        # predict head
        train_pred_mask = ~np.isnan(predict_arr[: fold.val_lo])
        val_pred_mask = ~np.isnan(predict_arr[fold.val_lo : fold.test_lo])

        y_train_pred = predict_arr[: fold.val_lo][core_train & train_pred_mask].astype(int)
        X_train_pred = X_train_raw[core_train & train_pred_mask]
        y_val_pred = predict_arr[fold.val_lo : fold.test_lo][core_val & val_pred_mask].astype(int)
        X_val_pred = X_val_raw[core_val & val_pred_mask]
        val_ku_pred = val_ku_raw[core_val & val_pred_mask]

        test_pred_mask = ~np.isnan(predict_arr[fold.test_lo : fold.test_hi])
        combined_test_mask = core_test & test_pred_mask
        y_test_pred = predict_arr[fold.test_lo : fold.test_hi][combined_test_mask].astype(int)
        X_test_pred = X_test_raw[combined_test_mask]
        test_weeks_pred = [w for w, ok in zip(test_weeks_all, combined_test_mask) if ok]

        if len(y_train_pred) < 20 or len(np.unique(y_train_pred)) < 2:
            continue

        pipe_p = make_pipeline_fn()
        pipe_p.fit(X_train_pred, y_train_pred)

        if len(y_val_pred) < 4 or len(np.unique(y_val_pred)) < 2:
            if use_logit:
                test_raw_p = pipe_p.predict_proba(X_test_pred)[:, 1]
            else:
                test_raw_p = pipe_p.decision_function(X_test_pred)
            predict_folds.append(
                FoldResult(
                    fold_year=fold.test_year,
                    head="predict",
                    weeks=test_weeks_pred,
                    true_labels=y_test_pred,
                    proba=test_raw_p,
                    alerts=(test_raw_p >= 0.0 if not use_logit else test_raw_p >= 0.5).astype(int),
                    threshold=0.0 if not use_logit else 0.5,
                    platt_a=1.0,
                    platt_b=0.0,
                    score_status="single_class_val",
                )
            )
            continue

        if use_logit:
            val_raw_p = pipe_p.predict_proba(X_val_pred)[:, 1]
        else:
            val_raw_p = pipe_p.decision_function(X_val_pred)
        platt_p = PlattCalibrator(use_logit=use_logit).fit(val_raw_p, y_val_pred)
        val_cal_p = platt_p.transform(val_raw_p)
        thr_p = choose_threshold(y_val_pred, val_cal_p, FAR_BUDGET, excluded_mask=val_ku_pred)

        if use_logit:
            test_raw_p = pipe_p.predict_proba(X_test_pred)[:, 1]
        else:
            test_raw_p = pipe_p.decision_function(X_test_pred)
        test_cal_p = platt_p.transform(test_raw_p)
        test_alerts_p = (test_cal_p >= thr_p).astype(int)

        predict_folds.append(
            FoldResult(
                fold_year=fold.test_year,
                head="predict",
                weeks=test_weeks_pred,
                true_labels=y_test_pred,
                proba=test_cal_p,
                alerts=test_alerts_p,
                threshold=thr_p,
                platt_a=platt_p.a,  # type: ignore[arg-type]
                platt_b=platt_p.b,  # type: ignore[arg-type]
            )
        )

    # ONNX gate: 500 synthetic rows with 15% NaN.
    # Use deepcopy with n_jobs=1 for determinism; training pipeline stays parallel.
    if detect_folds and last_pipe_d is not None:
        gate_rng = np.random.default_rng(42)
        X_gate = gate_rng.standard_normal((500, n_features)).astype(np.float32)
        X_gate[gate_rng.random((500, n_features)) < 0.15] = np.nan
        pipe_gate_onnx = copy.deepcopy(last_pipe_d)
        _set_njobs_1(pipe_gate_onnx)
        gate = check_onnx_gate(name, pipe_gate_onnx, last_platt_d, X_gate, n_features)
    else:
        gate = OnnxGateResult(
            passed=False, max_diff=None, error="No folds completed for gate test."
        )

    return CandidateResult(
        name=name, detect_folds=detect_folds, predict_folds=predict_folds, onnx_gate=gate
    )


# ── Pooled metrics ────────────────────────────────────────────────────────────


def compute_pooled_metrics(
    detect_folds: list[FoldResult],
    episodes: list[Episode],
    all_dev_weeks: list[date],
    calm_dev_years: list[int],
) -> dict:
    """Compute pooled row + event metrics from all detect folds."""
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    from sklearn.metrics import log_loss as sklearn_log_loss

    y_true = np.concatenate([fr.true_labels for fr in detect_folds])
    y_proba = np.concatenate([fr.proba for fr in detect_folds])
    y_alert = np.concatenate([fr.alerts for fr in detect_folds])
    all_alert_set = {w for fr in detect_folds for w, a in zip(fr.weeks, fr.alerts) if a == 1}

    row = row_metrics(y_true, y_alert)
    _, _, ef1 = event_precision_recall_f1(episodes, all_alert_set, sorted(set(all_dev_weeks)))
    ep_results = detect_event_results(episodes, all_alert_set)

    fa_runs = false_alarm_runs(
        sorted(all_alert_set & set(all_dev_weeks)),
        all_dev_weeks,
        episodes,
        grace_weeks=GRACE_WEEKS,
        known_unlabeled=KNOWN_UNLABELED,
    )
    fa_real = [r for r in fa_runs if not r.in_known_unlabeled]
    fa_calm = [r for r in fa_real if r.year in calm_dev_years]
    fa_per_calm = len(fa_calm) / max(len(calm_dev_years), 1)

    prob_m: dict | None = None
    if len(np.unique(y_true)) >= 2:
        prob_m = {
            "pr_auc": float(average_precision_score(y_true, y_proba)),
            "brier": float(brier_score_loss(y_true, y_proba)),
            "log_loss": float(sklearn_log_loss(y_true, y_proba)),
            "roc_auc": float(roc_auc_score(y_true, y_proba)),
        }

    return {
        "row": row,
        "event_f1": ef1,
        "episode_hits": sum(r.hit for r in ep_results),
        "episodes_total": len(episodes),
        "fa_calm_runs": len(fa_calm),
        "fa_total_runs": len(fa_real),
        "fa_per_calm": fa_per_calm,
        "calm_years": calm_dev_years,
        "prob_metrics": prob_m,
        "ep_results": ep_results,
        "log_loss": prob_m["log_loss"] if prob_m else float("nan"),
    }


# ── Case studies ──────────────────────────────────────────────────────────────


def build_case_study(
    candidate_name: str,
    detect_folds: list[FoldResult],
    year_ranges: list[tuple[int, int]],
) -> list[dict]:
    """Return week-by-week rows for the requested year ranges."""
    rows = []
    for fr in detect_folds:
        for w, p, a, y in zip(fr.weeks, fr.proba, fr.alerts, fr.true_labels):
            if any(start <= w.year <= end for start, end in year_ranges):
                rows.append(
                    {
                        "week": w,
                        "proba": round(float(p), 4),
                        "alert": int(a),
                        "label": int(y),
                    }
                )
    return sorted(rows, key=lambda r: r["week"])


# ── Report writer ─────────────────────────────────────────────────────────────


def write_selection_report(
    out_path: Path,
    candidates: dict[str, CandidateResult],
    baselines: dict[str, CandidateResult],
    pooled: dict[str, dict],
    boot: dict[str, dict],
    episodes: list[Episode],
    selected_name: str,
    selection_reason: str,
    vix_level_ef1: float,
    old_ensemble_ef1: float,
    label_set: str,
    first_test_year: int,
    last_test_year: int,
    n_boot: int = N_BOOT,
) -> None:
    today = date.today().isoformat()
    all_results: dict[str, CandidateResult] = {**candidates, **baselines}

    lines: list[str] = [
        f"# Model Selection Report — {label_set} Development Folds ({first_test_year}–{last_test_year})",
        "",
        f"Generated: {today}  ",
        f"Label set: `{label_set}` · Horizon h = {HORIZON} · Merge gap = 2 · FAR budget = {FAR_BUDGET:.0%}  ",
        f"Development folds: test years {first_test_year}–{last_test_year} (expanding window, val ≥ 8 positives)  ",
        "",
        "> **Note:** The 2018–2021 confirmation window is held out and will be evaluated once",
        "> at the end of Phase 2 Step 5. All metrics here are development-only.",
        "",
        "---",
        "",
        "## Summary table",
        "",
        "| Model | Event F1 | Episode hits | Row F1 | PR AUC | Brier | FA calm | FA total | ONNX gate |",
        "|-------|---------|-------------|--------|--------|-------|---------|---------|-----------|",
    ]

    all_names = list(candidates.keys()) + list(baselines.keys())
    # Always show VIX and old baseline rows even if not evaluated
    for extra in ("vix_level", "vix_z52", "old_isoforest", "old_ensemble", "old_ensemble_native05"):
        if extra not in all_names:
            all_names.append(extra)
    for nm in all_names:
        if nm not in pooled:
            lines.append(f"| {nm} | not evaluated | — | — | — | — | — | — | — |")
            continue
        m = pooled[nm]
        gate = all_results.get(nm)
        gate_obj = gate.onnx_gate if gate else None
        gate_str = (
            "pass"
            if gate_obj and gate_obj.passed
            else (f"FAIL: {gate_obj.error}" if gate_obj and gate_obj.error else "n/a")
        )
        pm = m.get("prob_metrics") or {}
        pr_auc = f"{pm.get('pr_auc', float('nan')):.3f}" if pm else "—"
        brier = f"{pm.get('brier', float('nan')):.3f}" if pm else "—"
        hits = f"{m['episode_hits']}/{m['episodes_total']}"
        row_f1 = m["row"]["f1"]
        fa_calm = m.get("fa_calm_runs", "—")
        fa_total = m.get("fa_total_runs", "—")
        lines.append(
            f"| {nm} | {m['event_f1']:.3f} | {hits} | {row_f1:.3f}"
            f" | {pr_auc} | {brier} | {fa_calm} | {fa_total} | {gate_str} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Per-episode hit table",
        "",
        "Candidates only (detect head, merge_gap=2, grace=±2 weeks).",
        "",
    ]

    if episodes:
        ep_table_names = list(candidates.keys()) + [
            n for n in ("vix_level", "old_ensemble") if n in pooled
        ]
        header_cols = ["#", "Start", "End", "Wks"] + ep_table_names
        lines.append("| " + " | ".join(header_cols) + " |")
        lines.append("|" + "---|" * len(header_cols))
        for i, ep in enumerate(episodes, 1):
            row_parts = [str(i), str(ep.start), str(ep.end), str(ep.length)]
            for nm in ep_table_names:
                if nm not in pooled:
                    row_parts.append("—")
                    continue
                ep_results = pooled[nm].get("ep_results", [])
                hit = ep_results[i - 1].hit if i - 1 < len(ep_results) else False
                row_parts.append("Y" if hit else "N")
            lines.append("| " + " | ".join(row_parts) + " |")

    # ── Predict-head section ─────────────────────────────────────────────────
    pred_names = [nm for nm in list(candidates.keys()) + ["vix_level"] if nm in all_results]

    def _pred_row_m(nm: str) -> dict | None:
        cr = all_results.get(nm)
        if not cr or not cr.predict_folds:
            return None
        yt = np.concatenate([fr.true_labels for fr in cr.predict_folds])
        ya = np.concatenate([fr.alerts for fr in cr.predict_folds])
        return row_metrics(yt, ya) if len(yt) > 0 else None

    lines += [
        "",
        "---",
        "",
        f"## Predict-head metrics (h={HORIZON} weeks)",
        "",
        "Row metrics for the h=4 advance-warning head (calm-at-t evaluation).",
        "",
        "| Model | Precision | Recall | F1 |",
        "|-------|-----------|--------|----|",
    ]
    for nm in pred_names:
        rm = _pred_row_m(nm)
        if rm is None:
            lines.append(f"| {nm} | — | — | — |")
        else:
            lines.append(f"| {nm} | {rm['precision']:.3f} | {rm['recall']:.3f} | {rm['f1']:.3f} |")

    # Per-episode advance warning: alert within h weeks before onset from predict head
    lines += [
        "",
        "### Advance warning (predict-head alert ≤ h weeks before episode onset)",
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
                # calm-at-t alert set: predict head fires AND current label is calm
                all_detect_weeks: dict[date, int] = {}
                for cr2 in all_results.values():
                    for fr in cr2.detect_folds:
                        for w, lbl in zip(fr.weeks, fr.true_labels):
                            all_detect_weeks[w] = int(lbl)
                calm_alerts = {
                    w
                    for fr in cr.predict_folds
                    for w, a in zip(fr.weeks, fr.alerts)
                    if a == 1 and all_detect_weeks.get(w, 1) == 0
                }
                ep_results = predict_event_results([ep], calm_alerts, HORIZON)
                r = ep_results[0]
                if r.hit and r.weeks_into_episode is not None:
                    row_parts.append(f"Y (+{r.weeks_into_episode}w)")
                else:
                    row_parts.append("N")
            lines.append("| " + " | ".join(row_parts) + " |")

    def _fmt_ci(ci: list | None) -> str:
        return f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "—"

    lines += [
        "",
        "---",
        "",
        f"## Bootstrap intervals (detect head, {n_boot} resamples of test years)",
        "",
        "| Model | Event F1 95% CI | Row F1 95% CI | PR AUC 95% CI | Brier 95% CI |",
        "|-------|----------------|--------------|--------------|-------------|",
    ]

    for nm in all_names:
        if nm not in boot:
            continue
        b = boot[nm]

        lines.append(
            f"| {nm} | {_fmt_ci(b.get('event_f1_ci95'))} | {_fmt_ci(b.get('row_f1_ci95'))}"
            f" | {_fmt_ci(b.get('pr_auc_ci95'))} | {_fmt_ci(b.get('brier_ci95'))} |"
        )

    lines += [
        "",
        "---",
        "",
        "## ONNX gate results",
        "",
        "| Model | Passed | Max diff (sklearn vs ONNX) | Notes |",
        "|-------|--------|---------------------------|-------|",
    ]

    for nm in candidates:
        gate = candidates[nm].onnx_gate
        passed = "yes" if gate.passed else "no"
        diff = f"{gate.max_diff:.2e}" if gate.max_diff is not None else "—"
        notes = gate.error or ""
        lines.append(f"| {nm} | {passed} | {diff} | {notes} |")

    lines += [
        "",
        "---",
        "",
        "## Selection result",
        "",
    ]

    def _paired_diff_ci(
        boot_a: list[float], boot_b: list[float]
    ) -> tuple[list[float] | None, bool]:
        if not boot_a or not boot_b or len(boot_a) != len(boot_b):
            return None, False
        diffs = np.asarray(boot_a) - np.asarray(boot_b)
        ci = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
        distinguishable = ci[0] > 0 or ci[1] < 0
        return ci, distinguishable

    if selected_name:
        winner_ef1 = pooled.get(selected_name, {}).get("event_f1", float("nan"))
        lines.append(f"**Selected model: `{selected_name}`**")
        lines.append("")
        lines.append(f"Reason: {selection_reason}")
        lines.append("")

        sel_ci = boot.get(selected_name, {}).get("event_f1_ci95")
        sel_boot_vals = boot.get(selected_name, {}).get("_boot_event_f1", [])
        sel_str = f"{winner_ef1:.3f} {_fmt_ci(sel_ci)}"

        # vs VIX level
        vix_ci = boot.get("vix_level", {}).get("event_f1_ci95")
        vix_boot_vals = boot.get("vix_level", {}).get("_boot_event_f1", [])
        diff_ci_vix, dist_vix = _paired_diff_ci(sel_boot_vals, vix_boot_vals)
        vix_str = f"{vix_level_ef1:.3f} {_fmt_ci(vix_ci)}"
        lines.append(
            f"- vs VIX level: selected {sel_str} · VIX {vix_str}"
            f" · paired diff {_fmt_ci(diff_ci_vix)}"
            f" · statistically distinguishable: **{'YES' if dist_vix else 'NO'}**"
        )
        if not dist_vix and winner_ef1 <= vix_level_ef1:
            lines.append("")
            lines.append(
                "> No candidate beat the VIX level rule on event F1. "
                "Stop for a decision (plan 2.4)."
            )

        # vs old ensemble
        ens_ci = boot.get("old_ensemble", {}).get("event_f1_ci95")
        ens_boot_vals = boot.get("old_ensemble", {}).get("_boot_event_f1", [])
        diff_ci_ens, dist_ens = _paired_diff_ci(sel_boot_vals, ens_boot_vals)
        ens_str = f"{old_ensemble_ef1:.3f} {_fmt_ci(ens_ci)}"
        lines.append(
            f"- vs old ensemble: selected {sel_str} · old ens {ens_str}"
            f" · paired diff {_fmt_ci(diff_ci_ens)}"
            f" · statistically distinguishable: **{'YES' if dist_ens else 'NO'}**"
        )
    else:
        lines.append("**No model selected.** " + selection_reason)

    # Fix 4: missing data table (rows dropped per fold)
    lines += [
        "",
        "---",
        "",
        "## Missing data per fold",
        "",
        "Rows excluded where any core feature (VIX, S&P 500, US 10y yield) is NaN.",
        "",
        "| Model | Fold year | Rows dropped |",
        "|-------|-----------|-------------|",
    ]
    for nm, cr in candidates.items():
        for fr in cr.detect_folds:
            lines.append(f"| {nm} | {fr.fold_year} | {fr.rows_dropped} |")

    # Fix 8: predict-head fallback footnote
    fallback_folds: dict[str, list[int]] = {}
    for nm, cr in candidates.items():
        fb_years = [
            fr.fold_year for fr in cr.predict_folds if fr.score_status == "single_class_val"
        ]
        if fb_years:
            fallback_folds[nm] = fb_years
    if fallback_folds:
        lines.append("")
        lines.append("**Predict-head fallback** (single-class val → threshold=0.5, no Platt):")
        for nm, years in fallback_folds.items():
            lines.append(f"- {nm}: fold years {years}")

    lines += [
        "",
        "---",
        "",
        "## Case studies",
        "",
        "### 2008 (Financial crisis)",
        "",
        "| Week | Prob | Alert | Label |",
        "|------|------|-------|-------|",
    ]

    if selected_name and selected_name in candidates:
        rows_2008 = build_case_study(
            selected_name, candidates[selected_name].detect_folds, [(2008, 2008)]
        )
        for r in rows_2008:
            lines.append(f"| {r['week']} | {r['proba']} | {r['alert']} | {r['label']} |")

    lines += [
        "",
        "### 2011–2012 (Euro debt crisis)",
        "",
        "| Week | Prob | Alert | Label |",
        "|------|------|-------|-------|",
    ]

    if selected_name and selected_name in candidates:
        rows_1112 = build_case_study(
            selected_name, candidates[selected_name].detect_folds, [(2011, 2012)]
        )
        for r in rows_1112:
            lines.append(f"| {r['week']} | {r['proba']} | {r['alert']} | {r['label']} |")

    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ── CLI entry point ───────────────────────────────────────────────────────────


def run_selection(
    session,
    out_dir: Path,
    label_set: str = "ews_v1",
    first_test_year: int = DEV_FIRST_YEAR,
    last_test_year: int = DEV_LAST_YEAR,
    n_boot: int = N_BOOT,
    csv_path: Path | None = None,
    skip_bootstrap: bool = False,
) -> int:
    """Load data, run all candidates, apply selection rule, write report."""
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

    # Build features from DB (point-in-time correct)
    feature_df = build_from_db(session, SPEC_V1, weeks, as_of=None)

    vix_close = weekly_close(session, "vix", weeks)
    if np.all(np.isnan(vix_close)):
        raise RuntimeError(
            "VIX weekly close is entirely NaN — cannot compute VIX baselines. "
            "Ensure the 'vix' instrument has price rows in the DB."
        )

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

    # Dev episodes
    ep2 = extract_episodes(labels_df, merge_gap=2)
    dev_start = date(first_test_year, 1, 1)
    dev_end = date(last_test_year, 12, 31)
    episodes = [e for e in ep2 if dev_start <= e.start <= dev_end]
    dev_weeks = [w for w in weeks if dev_start <= w <= dev_end]

    # Calm dev years
    ep_years: set[int] = set()
    from datetime import timedelta

    for ep in episodes:
        d = ep.start
        while d <= ep.end:
            ep_years.add(d.year)
            d += timedelta(weeks=1)
    calm_years = [y for y in range(first_test_year, last_test_year + 1) if y not in ep_years]

    # Evaluate candidates (gate sample is built inside evaluate_candidate — Fix 3)
    candidates: dict[str, CandidateResult] = {}
    for cname, factory in CANDIDATES.items():
        print(f"Evaluating candidate: {cname} ...")
        candidates[cname] = evaluate_candidate(
            cname,
            factory,
            feature_df,
            labels_df,
            folds,
            ku_mask,
        )

    # Compute pooled metrics for candidates
    pooled: dict[str, dict] = {}
    for cname, cr in candidates.items():
        pooled[cname] = compute_pooled_metrics(cr.detect_folds, episodes, dev_weeks, calm_years)

    # VIX baselines — same per-fold Platt + choose_threshold policy as candidates
    vix_baselines = _evaluate_vix_baselines(labels_df, folds, vix_close, ku_mask)
    for vname, vr in vix_baselines.items():
        if vr.detect_folds:
            pooled[vname] = compute_pooled_metrics(vr.detect_folds, episodes, dev_weeks, calm_years)

    # Old baselines (warn and skip gracefully if CSV unavailable)
    old_baselines: dict[str, CandidateResult] = {}
    if csv_path is None or not csv_path.exists():
        print("Old CSV baselines not evaluated (--csv not provided or file not found).")
    else:
        old_baselines = _evaluate_old_baselines(labels_df, folds, csv_path, ku_mask)
        for bname, br in old_baselines.items():
            if br.detect_folds:
                pooled[bname] = compute_pooled_metrics(
                    br.detect_folds, episodes, dev_weeks, calm_years
                )

    all_baselines = {**vix_baselines, **old_baselines}

    # Bootstrap
    boot: dict[str, dict] = {}
    if not skip_bootstrap:
        for cname, cr in candidates.items():
            print(f"Bootstrapping {cname} ...")
            boot[cname] = year_block_bootstrap(cr.detect_folds, episodes, dev_weeks, n_boot=n_boot)
        for bname, br in all_baselines.items():
            if br.detect_folds:
                boot[bname] = year_block_bootstrap(
                    br.detect_folds, episodes, dev_weeks, n_boot=n_boot
                )
    else:
        for nm in list(candidates.keys()) + list(all_baselines.keys()):
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

    # Selection
    onnx_gates = {nm: cr.onnx_gate for nm, cr in candidates.items()}
    sel_metrics = {nm: pooled[nm] for nm in candidates if nm in pooled}
    sel_boot = {nm: boot[nm] for nm in candidates if nm in boot}
    selected_name, selection_reason = apply_selection_rule(sel_metrics, sel_boot, onnx_gates)

    vix_level_ef1 = pooled.get("vix_level", {}).get("event_f1", float("nan"))
    if math.isnan(vix_level_ef1):
        raise RuntimeError(
            "VIX level event F1 could not be computed. "
            "Check that the 'vix' instrument has sufficient price rows."
        )
    old_ensemble_ef1 = pooled.get("old_ensemble", {}).get("event_f1", float("nan"))

    today = date.today().isoformat()
    out_path = out_dir / f"{today}-model-selection.md"
    write_selection_report(
        out_path,
        candidates,
        all_baselines,
        pooled,
        boot,
        episodes,
        selected_name,
        selection_reason,
        vix_level_ef1,
        old_ensemble_ef1,
        label_set,
        first_test_year,
        last_test_year,
        n_boot=n_boot,
    )
    print(f"model-selection: {selected_name or 'none'} → {out_path}")
    return 0


def build_export_df(
    hgb_result: CandidateResult,
    vix_result: CandidateResult,
    all_weeks: list[date],
    vix_close: np.ndarray,
    ku_mask: np.ndarray,
    episodes: list[Episode],
) -> pd.DataFrame:
    """Assemble per-dev-week DataFrame from hgb and vix_level CandidateResults.

    One row per dev week that appears in at least one detect fold. Weeks dropped
    by the core-feature mask (hgb only) still appear when covered by vix_level.
    Columns with no data for a given week are left as NaN / blank.
    """
    from datetime import timedelta

    week_to_ku = {w: bool(ku_mask[i]) for i, w in enumerate(all_weeks)}
    week_to_vix = {w: float(vix_close[i]) for i, w in enumerate(all_weeks)}

    week_to_ep: dict[date, int] = {}
    for ep_idx, ep in enumerate(episodes, 1):
        d = ep.start
        while d <= ep.end:
            week_to_ep[d] = ep_idx
            d += timedelta(weeks=1)

    rows: dict[date, dict] = {}

    def _ensure(w: date, fold_year: int, label: int) -> None:
        if w not in rows:
            rows[w] = {
                "week": w,
                "fold_year": fold_year,
                "label": label,
                "episode_id": week_to_ep.get(w, None),
                "known_unlabeled": int(week_to_ku.get(w, False)),
                "vix_close": week_to_vix.get(w, float("nan")),
                "hgb_prob": float("nan"),
                "hgb_threshold": float("nan"),
                "hgb_alert": None,
                "vix_level_prob": float("nan"),
                "vix_level_threshold": float("nan"),
                "vix_level_alert": None,
                "hgb_predict_prob": float("nan"),
                "hgb_predict_alert": None,
                "vix_predict_alert": None,
            }

    for fr in hgb_result.detect_folds:
        thr = fr.threshold
        for w, lbl, prob, alert in zip(fr.weeks, fr.true_labels, fr.proba, fr.alerts):
            _ensure(w, fr.fold_year, int(lbl))
            rows[w].update(hgb_prob=float(prob), hgb_threshold=float(thr), hgb_alert=int(alert))

    for fr in vix_result.detect_folds:
        thr = fr.threshold
        for w, lbl, prob, alert in zip(fr.weeks, fr.true_labels, fr.proba, fr.alerts):
            _ensure(w, fr.fold_year, int(lbl))
            rows[w].update(
                vix_level_prob=float(prob),
                vix_level_threshold=float(thr),
                vix_level_alert=int(alert),
            )

    for fr in hgb_result.predict_folds:
        for w, prob, alert in zip(fr.weeks, fr.proba, fr.alerts):
            if w in rows:
                rows[w].update(hgb_predict_prob=float(prob), hgb_predict_alert=int(alert))

    for fr in vix_result.predict_folds:
        for w, alert in zip(fr.weeks, fr.alerts):
            if w in rows:
                rows[w]["vix_predict_alert"] = int(alert)

    col_order = [
        "week",
        "fold_year",
        "label",
        "episode_id",
        "known_unlabeled",
        "vix_close",
        "hgb_prob",
        "hgb_threshold",
        "hgb_alert",
        "vix_level_prob",
        "vix_level_threshold",
        "vix_level_alert",
        "hgb_predict_prob",
        "hgb_predict_alert",
        "vix_predict_alert",
    ]
    df = pd.DataFrame(sorted(rows.values(), key=lambda r: r["week"]))
    return df[col_order]


def run_export_dev_predictions(
    session,
    out_path: Path,
    label_set: str = "ews_v1",
    first_test_year: int = DEV_FIRST_YEAR,
    last_test_year: int = DEV_LAST_YEAR,
) -> int:
    """Rerun hgb and vix_level on dev folds; write one row per dev week to out_path.

    Uses the exact same code path, seeds, and threshold policy as run_selection.
    Read-only: does not modify the DB or any report file.
    """
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

    print("export-dev-predictions: evaluating hgb ...")
    hgb_result = evaluate_candidate("hgb", CANDIDATES["hgb"], feature_df, labels_df, folds, ku_mask)
    print("export-dev-predictions: evaluating vix_level ...")
    vix_baselines = _evaluate_vix_baselines(labels_df, folds, vix_close, ku_mask)
    vix_result = vix_baselines["vix_level"]

    df = build_export_df(hgb_result, vix_result, weeks, vix_close, ku_mask, episodes)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    n_hgb = int((df["hgb_alert"] == 1).sum())
    n_vix = int((df["vix_level_alert"] == 1).sum())
    print(
        f"export-dev-predictions: {len(df)} dev weeks"
        f" | hgb alerts={n_hgb} vix_level alerts={n_vix}"
        f" → {out_path}"
    )
    return 0


def _evaluate_vix_baselines(
    labels_df: pd.DataFrame,
    folds: list[FoldSpec],
    vix_close: np.ndarray,
    ku_mask: np.ndarray,
) -> dict[str, CandidateResult]:
    """Evaluate vix_level and vix_z52 on the same dev folds as candidates.

    Each baseline uses per-fold Platt calibration + choose_threshold on the
    validation window (known-unlabeled excluded), same as the candidate policy.
    """
    from .eval_harness import choose_threshold, vix_rule
    from .eval_harness import vix_z52 as _vix_z52

    all_weeks: list[date] = list(labels_df["week"])
    detect_arr = labels_df["value"].values.astype(int)

    level_folds: list[FoldResult] = []
    z52_folds: list[FoldResult] = []

    for fold in folds:
        test_weeks = all_weeks[fold.test_lo : fold.test_hi]
        _guard_dev_only(test_weeks)

        val_detect = detect_arr[fold.val_lo : fold.test_lo]
        val_ku = ku_mask[fold.val_lo : fold.test_lo]
        test_detect = detect_arr[fold.test_lo : fold.test_hi]

        # ── vix_level ──────────────────────────────────────────────────────
        val_level = vix_close[fold.val_lo : fold.test_lo]
        test_level = vix_close[fold.test_lo : fold.test_hi]
        thr_level = choose_threshold(val_detect, val_level, FAR_BUDGET, excluded_mask=val_ku)
        test_alerts_level = vix_rule(test_level, thr_level)
        # Soft probability: sigmoid centered at threshold → values in (0,1)
        # so brier_score_loss and log_loss accept them.  The scale factor of 1
        # means the sigmoid transition spans roughly ±5 VIX points.
        test_proba_level = 1.0 / (1.0 + np.exp(-(test_level - thr_level)))
        level_folds.append(
            FoldResult(
                fold_year=fold.test_year,
                head="detect",
                weeks=test_weeks,
                true_labels=test_detect,
                proba=test_proba_level,
                alerts=test_alerts_level,
                threshold=thr_level,
                platt_a=1.0,
                platt_b=0.0,
            )
        )

        # ── vix_z52 ────────────────────────────────────────────────────────
        # z-score needs the full history up to end of val window for warmup
        z_full = _vix_z52(vix_close[: fold.test_lo])
        val_z52 = z_full[fold.val_lo :]  # aligns with val window
        test_z_full = _vix_z52(vix_close[: fold.test_hi])
        test_z52 = test_z_full[fold.test_lo :]

        thr_z52 = choose_threshold(val_detect, val_z52, FAR_BUDGET, excluded_mask=val_ku)
        test_alerts_z52 = vix_rule(test_z52, thr_z52)
        # Same sigmoid soft probability centered at threshold
        test_proba_z52 = 1.0 / (1.0 + np.exp(-(test_z52 - thr_z52)))
        z52_folds.append(
            FoldResult(
                fold_year=fold.test_year,
                head="detect",
                weeks=test_weeks,
                true_labels=test_detect,
                proba=test_proba_z52,
                alerts=test_alerts_z52,
                threshold=thr_z52,
                platt_a=1.0,
                platt_b=0.0,
            )
        )

    # ── vix_level predict head ────────────────────────────────────────────────
    from .labels import build_targets

    targets = build_targets(labels_df, horizon=HORIZON)
    predict_arr_vix = targets["predict"].values

    level_predict_folds: list[FoldResult] = []
    for fold in folds:
        test_weeks = all_weeks[fold.test_lo : fold.test_hi]
        try:
            _guard_dev_only(test_weeks)
        except ValueError:
            continue

        val_detect = detect_arr[fold.val_lo : fold.test_lo]
        val_ku = ku_mask[fold.val_lo : fold.test_lo]
        val_level = vix_close[fold.val_lo : fold.test_lo]
        test_level = vix_close[fold.test_lo : fold.test_hi]

        val_pred_raw = predict_arr_vix[fold.val_lo : fold.test_lo]
        test_pred_raw = predict_arr_vix[fold.test_lo : fold.test_hi]
        val_pred_mask = ~np.isnan(val_pred_raw)
        test_pred_mask = ~np.isnan(test_pred_raw)

        y_val_pred = val_pred_raw[val_pred_mask].astype(int)
        if len(y_val_pred) < 4 or len(np.unique(y_val_pred)) < 2:
            continue

        val_level_pred = val_level[val_pred_mask]
        val_ku_pred = val_ku[val_pred_mask]
        test_level_pred = test_level[test_pred_mask]
        y_test_pred = test_pred_raw[test_pred_mask].astype(int)
        test_weeks_pred = [w for w, ok in zip(test_weeks, test_pred_mask) if ok]

        thr_pred = choose_threshold(
            y_val_pred, val_level_pred, FAR_BUDGET, excluded_mask=val_ku_pred
        )
        test_alerts_pred = vix_rule(test_level_pred, thr_pred)
        test_proba_pred = 1.0 / (1.0 + np.exp(-(test_level_pred - thr_pred)))

        level_predict_folds.append(
            FoldResult(
                fold_year=fold.test_year,
                head="predict",
                weeks=test_weeks_pred,
                true_labels=y_test_pred,
                proba=test_proba_pred,
                alerts=test_alerts_pred,
                threshold=thr_pred,
                platt_a=1.0,
                platt_b=0.0,
            )
        )

    no_gate = OnnxGateResult(passed=False, max_diff=None, error="VIX rule — no ONNX export")
    return {
        "vix_level": CandidateResult(
            name="vix_level",
            detect_folds=level_folds,
            predict_folds=level_predict_folds,
            onnx_gate=no_gate,
        ),
        "vix_z52": CandidateResult(
            name="vix_z52", detect_folds=z52_folds, predict_folds=[], onnx_gate=no_gate
        ),
    }


def _evaluate_old_baselines(
    labels_df: pd.DataFrame,
    folds: list[FoldSpec],
    csv_path: Path,
    ku_mask: np.ndarray,
) -> dict[str, CandidateResult]:
    """Evaluate old 3-feature IsoForest and old ensemble on the same dev folds.

    Both use the same Platt + choose_threshold policy as the main candidates.
    old_ensemble_native05 uses the raw 0.5 threshold without outer Platt (oracle comparison).
    """
    try:
        from .data import FEATURE_COLS, engineer_features, load_data
    except ImportError:
        return {}

    OLD_ISOFOREST_FEATURES = ["VIX", "USD_Over_100", "XAU BGNL_cross"]

    try:
        csv_df = load_data(str(csv_path))
        csv_df = engineer_features(csv_df)
    except Exception:
        return {}

    all_weeks: list[date] = list(labels_df["week"])
    results: dict[str, CandidateResult] = {}
    _gate = OnnxGateResult(passed=False, max_diff=None, error="Baseline — ONNX gate not applied.")

    # ── old_isoforest ──────────────────────────────────────────────────────────
    iso_folds: list[FoldResult] = []
    for fold in folds:
        test_weeks = all_weeks[fold.test_lo : fold.test_hi]
        try:
            _guard_dev_only(test_weeks)
        except ValueError:
            continue

        train_weeks_set = set(all_weeks[: fold.val_lo])
        val_weeks_set = set(all_weeks[fold.val_lo : fold.test_lo])
        test_weeks_set = set(test_weeks)

        train_csv = csv_df[csv_df["Data"].dt.date.isin(train_weeks_set)].sort_values("Data")
        val_csv = csv_df[csv_df["Data"].dt.date.isin(val_weeks_set)].sort_values("Data")
        test_csv = csv_df[csv_df["Data"].dt.date.isin(test_weeks_set)].sort_values("Data")

        if len(train_csv) < 20 or len(val_csv) == 0 or len(test_csv) == 0:
            continue
        missing = [c for c in OLD_ISOFOREST_FEATURES if c not in train_csv.columns]
        if missing:
            continue

        X_train_iso = train_csv[OLD_ISOFOREST_FEATURES].values
        X_val_iso = val_csv[OLD_ISOFOREST_FEATURES].values
        y_val_iso = val_csv["Y"].values.astype(int)
        X_test_iso = test_csv[OLD_ISOFOREST_FEATURES].values
        y_test_iso = test_csv["Y"].values.astype(int)
        test_weeks_aligned: list[date] = [d.date() for d in test_csv["Data"]]
        val_ku_iso = ku_mask[fold.val_lo : fold.test_lo][
            [i for i, w in enumerate(all_weeks[fold.val_lo : fold.test_lo]) if w in val_weeks_set]
        ]

        iso = IsolationForest(n_estimators=500, contamination="auto", random_state=42)
        iso.fit(X_train_iso)

        # Negate: higher = more anomalous (consistent with anomaly semantics)
        val_scores_iso = -iso.decision_function(X_val_iso)
        platt_iso = PlattCalibrator().fit(val_scores_iso, y_val_iso)
        val_cal_iso = platt_iso.transform(val_scores_iso)
        thr_iso = choose_threshold(y_val_iso, val_cal_iso, FAR_BUDGET, excluded_mask=val_ku_iso)

        test_scores_iso = -iso.decision_function(X_test_iso)
        test_cal_iso = platt_iso.transform(test_scores_iso)
        test_alerts_iso = (test_cal_iso >= thr_iso).astype(int)

        iso_folds.append(
            FoldResult(
                fold_year=fold.test_year,
                head="detect",
                weeks=test_weeks_aligned,
                true_labels=y_test_iso,
                proba=test_cal_iso,
                alerts=test_alerts_iso,
                threshold=thr_iso,
                platt_a=platt_iso.a,  # type: ignore[arg-type]
                platt_b=platt_iso.b,  # type: ignore[arg-type]
            )
        )
    results["old_isoforest"] = CandidateResult(
        name="old_isoforest", detect_folds=iso_folds, predict_folds=[], onnx_gate=_gate
    )

    # ── old_ensemble and old_ensemble_native05 ─────────────────────────────────
    ens_folds: list[FoldResult] = []
    ens_native_folds: list[FoldResult] = []

    for fold in folds:
        test_weeks = all_weeks[fold.test_lo : fold.test_hi]
        try:
            _guard_dev_only(test_weeks)
        except ValueError:
            continue

        train_weeks_set = set(all_weeks[: fold.val_lo])
        val_weeks_set = set(all_weeks[fold.val_lo : fold.test_lo])
        test_weeks_set = set(test_weeks)

        train_csv = csv_df[csv_df["Data"].dt.date.isin(train_weeks_set)].sort_values("Data")
        val_csv = csv_df[csv_df["Data"].dt.date.isin(val_weeks_set)].sort_values("Data")
        test_csv = csv_df[csv_df["Data"].dt.date.isin(test_weeks_set)].sort_values("Data")

        if len(train_csv) < 20 or len(val_csv) == 0 or len(test_csv) == 0:
            continue
        missing = [c for c in FEATURE_COLS if c not in train_csv.columns]
        if missing:
            continue

        y_train_ens = train_csv["Y"].values.astype(int)
        y_val_ens = val_csv["Y"].values.astype(int)
        y_test_ens = test_csv["Y"].values.astype(int)
        test_weeks_aligned2: list[date] = [d.date() for d in test_csv["Data"]]
        val_ku_ens = ku_mask[fold.val_lo : fold.test_lo][
            [i for i, w in enumerate(all_weeks[fold.val_lo : fold.test_lo]) if w in val_weeks_set]
        ]

        try:
            from .pipeline import AnomalyPipeline
        except ImportError:
            continue

        train_df = train_csv[FEATURE_COLS].copy()
        val_df = val_csv[FEATURE_COLS].copy()
        test_df = test_csv[FEATURE_COLS].copy()

        pipe_ens = AnomalyPipeline(calibration="sigmoid")
        pipe_ens.fit(train_df, pd.Series(y_train_ens), val_df, pd.Series(y_val_ens))

        test_proba_ens = pipe_ens.predict_proba(test_df)[:, 1]

        # native 0.5
        ens_native_folds.append(
            FoldResult(
                fold_year=fold.test_year,
                head="detect",
                weeks=test_weeks_aligned2,
                true_labels=y_test_ens,
                proba=test_proba_ens,
                alerts=(test_proba_ens >= 0.5).astype(int),
                threshold=0.5,
                platt_a=1.0,
                platt_b=0.0,
            )
        )

        # outer Platt + choose_threshold policy
        val_proba_ens = pipe_ens.predict_proba(val_df)[:, 1]
        platt_ens = PlattCalibrator().fit(val_proba_ens, y_val_ens)
        val_cal_ens = platt_ens.transform(val_proba_ens)
        thr_ens = choose_threshold(y_val_ens, val_cal_ens, FAR_BUDGET, excluded_mask=val_ku_ens)
        test_cal_ens = platt_ens.transform(test_proba_ens)
        test_alerts_ens = (test_cal_ens >= thr_ens).astype(int)

        ens_folds.append(
            FoldResult(
                fold_year=fold.test_year,
                head="detect",
                weeks=test_weeks_aligned2,
                true_labels=y_test_ens,
                proba=test_cal_ens,
                alerts=test_alerts_ens,
                threshold=thr_ens,
                platt_a=platt_ens.a,  # type: ignore[arg-type]
                platt_b=platt_ens.b,  # type: ignore[arg-type]
            )
        )

    results["old_ensemble"] = CandidateResult(
        name="old_ensemble", detect_folds=ens_folds, predict_folds=[], onnx_gate=_gate
    )
    results["old_ensemble_native05"] = CandidateResult(
        name="old_ensemble_native05",
        detect_folds=ens_native_folds,
        predict_folds=[],
        onnx_gate=_gate,
    )

    return results


# ── Calibration diagnostic ────────────────────────────────────────────────────


def collect_calibration_diagnostics(
    name: str,
    make_pipeline_fn: Callable[[], Pipeline],
    X_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    folds: list[FoldSpec],
    ku_mask: np.ndarray,
) -> list[dict]:
    """Re-run the detect-head fold loop and collect per-fold calibration stats.

    Mirrors evaluate_candidate's detect head exactly (same seeds, same core
    masking) so numbers are identical; returns raw-val-score statistics that
    FoldResult doesn't store, without touching evaluate_candidate at all.
    """
    all_weeks: list[date] = list(labels_df["week"])
    detect_arr = labels_df["value"].values.astype(int)
    X_all = X_df.values.astype(float) if hasattr(X_df, "values") else np.asarray(X_df)
    core_ok = _core_feature_mask(X_df, X_all)

    from .eval_harness import choose_threshold as _choose_thr

    rows: list[dict] = []
    for fold in folds:
        test_weeks_all = all_weeks[fold.test_lo : fold.test_hi]
        try:
            _guard_dev_only(test_weeks_all)
        except ValueError:
            continue

        val_start = all_weeks[fold.val_lo]
        val_end = all_weeks[fold.test_lo - 1]

        core_train = core_ok[: fold.val_lo]
        core_val = core_ok[fold.val_lo : fold.test_lo]
        core_test = core_ok[fold.test_lo : fold.test_hi]

        X_train = X_all[: fold.val_lo][core_train]
        y_train = detect_arr[: fold.val_lo][core_train]
        X_val = X_all[fold.val_lo : fold.test_lo][core_val]
        y_val = detect_arr[fold.val_lo : fold.test_lo][core_val]
        val_ku = ku_mask[fold.val_lo : fold.test_lo][core_val]
        X_test = X_all[fold.test_lo : fold.test_hi][core_test]

        if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
            continue
        if len(np.unique(y_train)) < 2:
            continue

        use_logit_diag = _CANDIDATE_USES_LOGIT.get(name, True)
        pipe = make_pipeline_fn()
        pipe.fit(X_train, y_train)

        if use_logit_diag:
            val_raw = pipe.predict_proba(X_val)[:, 1]
        else:
            val_raw = pipe.decision_function(X_val)
        platt = PlattCalibrator(use_logit=use_logit_diag).fit(val_raw, y_val)
        val_cal = platt.transform(val_raw)
        thr = _choose_thr(y_val, val_cal, FAR_BUDGET, excluded_mask=val_ku)

        if use_logit_diag:
            test_raw = pipe.predict_proba(X_test)[:, 1]
        else:
            test_raw = pipe.decision_function(X_test)
        test_cal = platt.transform(test_raw)

        # Achieved calm-week FAR on val (excluding KU, matching choose_threshold)
        calm_val = (y_val == 0) & (~val_ku.astype(bool))
        if calm_val.any():
            val_calm_far = float((val_cal[calm_val] >= thr).mean())
        else:
            val_calm_far = float("nan")

        rows.append(
            {
                "candidate": name,
                "fold_year": fold.test_year,
                "val_start": val_start,
                "val_end": val_end,
                "val_pos": int(y_val.sum()),
                "val_total": int(len(y_val)),
                "platt_a": float(platt.a),  # type: ignore[arg-type]
                "platt_b": float(platt.b),  # type: ignore[arg-type]
                "threshold": float(thr),
                "raw_val_min": float(np.nanmin(val_raw)),
                "raw_val_med": float(np.nanmedian(val_raw)),
                "raw_val_max": float(np.nanmax(val_raw)),
                "cal_test_min": float(np.nanmin(test_cal)),
                "cal_test_med": float(np.nanmedian(test_cal)),
                "cal_test_max": float(np.nanmax(test_cal)),
                "val_calm_far": val_calm_far,
            }
        )

    return rows


def write_calibration_diagnostic(
    out_path: Path,
    diag_rows: list[dict],
    old_ens_native_far_rows: list[dict],
) -> None:
    """Write docs/reports/<date>-calibration-diagnostic.md."""
    today = date.today().isoformat()

    def _f(v: float, fmt: str = ".4f") -> str:
        if math.isnan(v):
            return "—"
        if v == float("inf"):
            return "inf"
        return format(v, fmt)

    lines: list[str] = [
        "# Calibration Diagnostic",
        "",
        f"Generated: {today}  ",
        "Candidates: logreg, hgb, twostage  ",
        f"FAR budget = {FAR_BUDGET:.0%} · detect head · test years {DEV_FIRST_YEAR}–{DEV_LAST_YEAR}  ",
        "",
        "---",
        "",
        "## Per-fold calibration",
        "",
        "| Candidate | Fold | Val window | Val pos / total"
        " | Platt a | Platt b | Threshold"
        " | Raw val min | Raw val med | Raw val max"
        " | Cal test min | Cal test med | Cal test max"
        " | Val calm FAR |",
        "|-----------|------|-----------|----------------"
        "|---------|---------|----------"
        "|-------------|-------------|-------------"
        "|--------------|--------------|--------------|--------------|",
    ]

    for r in diag_rows:
        lines.append(
            f"| {r['candidate']} | {r['fold_year']}"
            f" | {r['val_start']}–{r['val_end']}"
            f" | {r['val_pos']} / {r['val_total']}"
            f" | {_f(r['platt_a'])} | {_f(r['platt_b'])} | {_f(r['threshold'])}"
            f" | {_f(r['raw_val_min'])} | {_f(r['raw_val_med'])} | {_f(r['raw_val_max'])}"
            f" | {_f(r['cal_test_min'])} | {_f(r['cal_test_med'])} | {_f(r['cal_test_max'])}"
            f" | {_f(r['val_calm_far'])} |"
        )

    lines += [
        "",
        "---",
        "",
        "## old_ensemble_native05 — calm-week FAR on test folds",
        "",
        "Threshold = 0.5 (native). Calm weeks exclude known-unlabeled windows.",
        "",
        "| Fold | Calm-week FAR (test) |",
        "|------|---------------------|",
    ]

    if old_ens_native_far_rows:
        for r in old_ens_native_far_rows:
            lines.append(f"| {r['fold_year']} | {_f(r['calm_test_far'])} |")
    else:
        lines.append("| — | old CSV not provided; run with --csv to populate |")

    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_calibration_diagnostic(
    session,
    out_dir: Path,
    label_set: str = "ews_v1",
    first_test_year: int = DEV_FIRST_YEAR,
    last_test_year: int = DEV_LAST_YEAR,
    csv_path: Path | None = None,
) -> int:
    """Load data, collect per-fold calibration stats for all candidates, write report."""
    from .eval_harness import KNOWN_UNLABELED, make_folds_v2
    from .features.build import build_from_db
    from .features.spec import SPEC_V1
    from .labels import load_labels

    labels_df = load_labels(session, label_set=label_set)
    if labels_df.empty:
        print(f"ERROR: no labels for {label_set!r}")
        return 1

    weeks: list[date] = list(labels_df["week"])
    feature_df = build_from_db(session, SPEC_V1, weeks, as_of=None)

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

    diag_rows: list[dict] = []
    for cname, factory in CANDIDATES.items():
        print(f"Calibration diagnostic: {cname} ...")
        diag_rows.extend(
            collect_calibration_diagnostics(cname, factory, feature_df, labels_df, folds, ku_mask)
        )

    # old_ensemble_native05 calm-week test FAR (requires old CSV)
    old_ens_native_far_rows: list[dict] = []
    if csv_path is not None and csv_path.exists():
        old_baselines = _evaluate_old_baselines(labels_df, folds, csv_path, ku_mask)
        cr = old_baselines.get("old_ensemble_native05")
        if cr is not None:
            week_to_ku: dict[date, bool] = {w: bool(ku_mask[i]) for i, w in enumerate(weeks)}
            for fr in cr.detect_folds:
                calm = np.array(
                    [
                        (int(y) == 0) and not week_to_ku.get(w, False)
                        for y, w in zip(fr.true_labels, fr.weeks)
                    ]
                )
                far = float(fr.alerts[calm].mean()) if calm.any() else float("nan")
                old_ens_native_far_rows.append({"fold_year": fr.fold_year, "calm_test_far": far})
    else:
        print("old_ensemble_native05 FAR skipped (--csv not provided or not found).")

    today = date.today().isoformat()
    out_path = out_dir / f"{today}-calibration-diagnostic.md"
    write_calibration_diagnostic(out_path, diag_rows, old_ens_native_far_rows)
    print(f"calibration-diagnostic → {out_path}")
    return 0
