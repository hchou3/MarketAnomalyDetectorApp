"""Tests for Phase 2 Step 2.4 — candidate model selection.

All tests use synthetic data; no DB or network required.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from market_anomaly_detector.calibration import PlattCalibrator
from market_anomaly_detector.episodes import Episode
from market_anomaly_detector.eval_harness import choose_threshold
from market_anomaly_detector.selection import (
    MedianImputeWithIndicator,
    OnnxGateResult,
    _guard_dev_only,
    apply_selection_rule,
    event_precision_recall_f1,
)

# ── helpers ───────────────────────────────────────────────────────────────────


def _weeks_range(start: date, n: int) -> list[date]:
    return [start + timedelta(weeks=i) for i in range(n)]


def _episode(start: date, length: int) -> Episode:
    end = start + timedelta(weeks=length - 1)
    return Episode(start=start, end=end, peak_week=start, length=length)


# ── PlattCalibrator ───────────────────────────────────────────────────────────


def test_platt_fit_apply():
    rng = np.random.default_rng(0)
    # Positives have scores near 2, negatives near -1
    scores_pos = rng.normal(2.0, 0.5, 40)
    scores_neg = rng.normal(-1.0, 0.5, 100)
    scores = np.concatenate([scores_pos, scores_neg])
    labels = np.concatenate([np.ones(40), np.zeros(100)])

    cal = PlattCalibrator()
    cal.fit(scores, labels)
    assert cal.a is not None
    assert cal.b is not None

    probs = cal.transform(scores)
    assert probs.shape == scores.shape
    assert (probs >= 0).all() and (probs <= 1).all()

    # High-score inputs should have higher probability than low-score
    assert cal.transform(np.array([2.0]))[0] > cal.transform(np.array([-1.0]))[0]


def test_platt_transform_before_fit_raises():
    cal = PlattCalibrator()
    with pytest.raises(RuntimeError, match="fit"):
        cal.transform(np.array([0.5]))


def test_platt_fit_transform_equiv():
    rng = np.random.default_rng(1)
    scores = rng.normal(0, 1, 50)
    labels = (scores > 0).astype(int)
    cal = PlattCalibrator()
    out1 = cal.fit_transform(scores, labels)
    out2 = cal.transform(scores)
    np.testing.assert_allclose(out1, out2, rtol=1e-10)


# ── Threshold from val rows only ──────────────────────────────────────────────


def test_platt_threshold_val_only():
    """Threshold must be chosen only from val rows, not test rows.

    Val scores are moderate; test contains extreme high scores that would
    raise the threshold if mistakenly included.
    """
    rng = np.random.default_rng(2)
    # Val: 40 calm (scores ~0.3), 10 anomaly (scores ~0.9)
    val_calm = rng.uniform(0.1, 0.5, 40)
    val_anom = rng.uniform(0.7, 0.95, 10)
    val_scores = np.concatenate([val_calm, val_anom])
    val_labels = np.concatenate([np.zeros(40), np.ones(10)])

    # Test: all calm but with very high scores (should NOT affect threshold)
    test_calm = rng.uniform(0.8, 0.99, 50)
    test_labels = np.zeros(50, dtype=int)

    thr_val_only = choose_threshold(val_labels, val_scores, far_budget=0.05)

    # If we mistakenly included test data, threshold would be much higher
    combined_scores = np.concatenate([val_scores, test_calm])
    combined_labels = np.concatenate([val_labels, test_labels])
    thr_combined = choose_threshold(combined_labels, combined_scores, far_budget=0.05)

    # Val-only threshold should be ≤ combined (test has higher calm scores → higher threshold)
    assert thr_val_only <= thr_combined, (
        f"Val-only threshold {thr_val_only:.4f} should be <= combined {thr_combined:.4f}"
    )
    # Sanity: val-only threshold is finite
    assert not math.isinf(thr_val_only)


# ── PlattCalibrator log-odds ──────────────────────────────────────────────────


def test_platt_log_odds_recovery():
    """Platt fitted on log-odds recovers a known (a, b) from noisy binary labels.

    Labels are drawn from sigmoid(a_true * logit(p_raw) + b_true), so the
    MLE estimate should converge to (a_true, b_true) with enough data.
    """
    rng = np.random.default_rng(99)
    n = 2000
    a_true, b_true = 2.0, -1.0

    # Raw scores as probabilities (skewed, range within (0,1))
    p_raw = rng.beta(2, 5, n)
    logit_raw = np.log(p_raw / (1 - p_raw))
    p_true = expit(a_true * logit_raw + b_true)
    labels = rng.binomial(1, p_true).astype(float)

    cal = PlattCalibrator().fit(p_raw, labels)

    assert abs(cal.a - a_true) < 0.4, f"a={cal.a:.3f} too far from {a_true}"
    assert abs(cal.b - b_true) < 0.4, f"b={cal.b:.3f} too far from {b_true}"
    probs = cal.transform(p_raw)
    assert probs.min() >= 0.0 and probs.max() <= 1.0


def test_platt_saturated_distribution_non_degenerate():
    """Saturated raw scores (most ≈ 0, a few ≈ 1) give non-degenerate calibrated range.

    HGB/stacking on an imbalanced dataset often saturates: negatives cluster near 0,
    positives near 1.  Fitting on log-odds handles this correctly; fitting on raw
    probabilities collapses to near-constant output.
    """
    rng = np.random.default_rng(42)
    n_neg, n_pos = 1800, 200

    # Saturated: negatives ≈ 0 (median ~0.004), positives ≈ 1
    neg_scores = rng.beta(0.5, 50, n_neg)
    pos_scores = rng.beta(50, 0.5, n_pos)
    raw = np.concatenate([neg_scores, pos_scores])
    labels = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])

    cal = PlattCalibrator().fit(raw, labels)
    probs = cal.transform(raw)

    assert probs.min() >= 0.0 and probs.max() <= 1.0
    assert probs.max() - probs.min() > 0.1, (
        f"Calibrated range too narrow: {probs.max() - probs.min():.4f}"
    )
    assert probs[labels == 1].mean() > probs[labels == 0].mean()


# ── Event precision / recall / F1 edge cases ─────────────────────────────────


def test_event_f1_no_alerts():
    ep = _episode(date(2010, 3, 1), 4)
    all_weeks = _weeks_range(date(2010, 1, 4), 52)
    prec, rec, f1 = event_precision_recall_f1([ep], set(), all_weeks)
    assert prec == 0.0
    assert rec == 0.0
    assert f1 == 0.0


def test_event_f1_no_episodes():
    all_weeks = _weeks_range(date(2010, 1, 4), 52)
    alert_set = {all_weeks[10], all_weeks[11]}
    prec, rec, f1 = event_precision_recall_f1([], alert_set, all_weeks)
    assert prec == 0.0
    assert rec == 0.0
    assert f1 == 0.0


def test_event_f1_perfect():
    ep = _episode(date(2010, 3, 1), 4)
    all_weeks = _weeks_range(date(2010, 1, 4), 52)
    # One alert run that lands exactly inside the episode
    alert_set = {ep.start}
    prec, rec, f1 = event_precision_recall_f1([ep], alert_set, all_weeks)
    assert rec == 1.0
    assert prec == 1.0
    assert abs(f1 - 1.0) < 1e-9


def test_event_f1_run_spanning_two_episodes():
    """One contiguous alert run that overlaps two distinct episodes.

    The run is counted as ONE run for precision but hits BOTH episodes for recall.
    Precision should be 1.0 (1 run, both episodes covered).
    Recall should be 1.0 (both episodes hit).
    """
    # Two episodes 6 weeks apart (so no merge at gap=2)
    ep1 = _episode(date(2010, 3, 1), 3)  # 2010-03-01 to 2010-03-15
    ep2 = _episode(date(2010, 4, 26), 3)  # 2010-04-26 to 2010-05-10

    # Build all_weeks covering both
    all_weeks = _weeks_range(date(2010, 1, 4), 30)

    # One alert run that spans both episodes (and the gap between them)
    alert_set: set[date] = set()
    d = ep1.start
    while d <= ep2.end:
        if d in set(all_weeks):
            alert_set.add(d)
        d += timedelta(weeks=1)

    prec, rec, f1 = event_precision_recall_f1([ep1, ep2], alert_set, all_weeks)
    # Both episodes hit → recall = 1.0
    assert rec == 1.0, f"Expected recall=1.0, got {rec}"
    # 1 run that hits both → precision = 1.0
    assert prec == 1.0, f"Expected precision=1.0, got {prec}"
    assert abs(f1 - 1.0) < 1e-9


def test_event_f1_false_alarm_run():
    """A run that misses all episodes counts as a FA → precision < 1."""
    ep = _episode(date(2010, 6, 1), 3)
    all_weeks = _weeks_range(date(2010, 1, 4), 52)
    # Alert run far from episode
    alert_set = {all_weeks[0], all_weeks[1]}  # January, episode is in June
    prec, rec, f1 = event_precision_recall_f1([ep], alert_set, all_weeks)
    assert rec == 0.0  # episode missed
    assert prec == 0.0  # run is a false alarm
    assert f1 == 0.0


# ── Guard function ────────────────────────────────────────────────────────────


def test_guard_dev_only_raises():
    bad_week = date(2018, 1, 2)
    with pytest.raises(ValueError, match="2018-01-01"):
        _guard_dev_only([date(2017, 12, 26), bad_week])


def test_guard_dev_only_passes():
    good_weeks = _weeks_range(date(2008, 1, 1), 52)
    _guard_dev_only(good_weeks)  # should not raise


def test_guard_dev_only_exact_cutoff_raises():
    with pytest.raises(ValueError):
        _guard_dev_only([date(2018, 1, 1)])


# ── Selection rule ────────────────────────────────────────────────────────────


def test_selection_rule_best_wins():
    """Simplest candidate with strictly best event F1 is selected."""
    metrics = {
        "logreg": {"event_f1": 0.50, "fa_per_calm": 1.0, "log_loss": 0.5},
        "hgb": {"event_f1": 0.65, "fa_per_calm": 0.5, "log_loss": 0.4},
        "twostage": {"event_f1": 0.60, "fa_per_calm": 0.5, "log_loss": 0.45},
    }
    boot = {
        "logreg": {"event_f1_se": 0.05, "_boot_event_f1": [0.50] * 100},
        "hgb": {"event_f1_se": 0.03, "_boot_event_f1": [0.65] * 100},
        "twostage": {"event_f1_se": 0.04, "_boot_event_f1": [0.60] * 100},
    }
    gates = {
        "logreg": OnnxGateResult(passed=True, max_diff=1e-7, error=None),
        "hgb": OnnxGateResult(passed=True, max_diff=1e-7, error=None),
        "twostage": OnnxGateResult(passed=True, max_diff=1e-7, error=None),
    }
    selected, reason = apply_selection_rule(metrics, boot, gates)
    assert selected == "hgb"


def test_selection_rule_se_tiebreak():
    """Simplest within 1 SE of best wins, even if not the absolute best."""
    # hgb is best (0.70) but logreg is within 1 SE (0.70 - 0.15 = 0.55, logreg=0.60)
    metrics = {
        "logreg": {"event_f1": 0.60, "fa_per_calm": 0.5, "log_loss": 0.4},
        "hgb": {"event_f1": 0.70, "fa_per_calm": 0.5, "log_loss": 0.45},
        "twostage": {"event_f1": 0.68, "fa_per_calm": 0.5, "log_loss": 0.44},
    }
    boot = {
        "logreg": {"event_f1_se": 0.05, "_boot_event_f1": [0.60] * 100},
        "hgb": {"event_f1_se": 0.15, "_boot_event_f1": [0.70] * 100},
        "twostage": {"event_f1_se": 0.06, "_boot_event_f1": [0.68] * 100},
    }
    gates = {
        "logreg": OnnxGateResult(passed=True, max_diff=1e-7, error=None),
        "hgb": OnnxGateResult(passed=True, max_diff=1e-7, error=None),
        "twostage": OnnxGateResult(passed=True, max_diff=1e-7, error=None),
    }
    selected, reason = apply_selection_rule(metrics, boot, gates)
    # logreg (0.60) is within 1 SE of hgb (0.70 - 0.15 = 0.55), and is simpler
    assert selected == "logreg"
    assert "simpler" in reason.lower() or "within" in reason.lower()


def test_selection_rule_onnx_gate_excludes():
    """Candidates failing ONNX gate cannot be selected."""
    metrics = {
        "logreg": {"event_f1": 0.40, "fa_per_calm": 1.0, "log_loss": 0.6},
        "hgb": {"event_f1": 0.80, "fa_per_calm": 0.5, "log_loss": 0.3},
    }
    boot = {
        "logreg": {"event_f1_se": 0.05, "_boot_event_f1": [0.40] * 100},
        "hgb": {"event_f1_se": 0.04, "_boot_event_f1": [0.80] * 100},
    }
    gates = {
        "logreg": OnnxGateResult(passed=True, max_diff=1e-7, error=None),
        "hgb": OnnxGateResult(
            passed=False, max_diff=None, error="Export failed: custom transformer"
        ),
    }
    selected, reason = apply_selection_rule(metrics, boot, gates)
    # hgb is better but failed gate → logreg wins
    assert selected == "logreg"


def test_selection_rule_no_eligible():
    gates = {
        "logreg": OnnxGateResult(passed=False, max_diff=None, error="fail"),
        "hgb": OnnxGateResult(passed=False, max_diff=None, error="fail"),
    }
    selected, reason = apply_selection_rule(
        {"logreg": {"event_f1": 0.5, "fa_per_calm": 0, "log_loss": 0.5}},
        {"logreg": {"event_f1_se": 0.05}},
        gates,
    )
    assert selected == ""


# ── ONNX parity ───────────────────────────────────────────────────────────────


def _make_synthetic_data(
    n: int = 500, n_features: int = 10, seed: int = 42, nan_frac: float = 0.15
):
    """Return (X, y) with ~nan_frac NaN entries (simulates real feature matrix)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    nan_mask = rng.random((n, n_features)) < nan_frac
    X[nan_mask] = np.nan
    y = (np.nan_to_num(X[:, 0]) + rng.standard_normal(n) * 0.5 > 0).astype(int)
    return X, y


try:
    import onnxruntime  # noqa: F401
    import skl2onnx  # noqa: F401

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


def _make_saturated_rows(
    n: int, n_features: int, seed: int, nan_frac: float = 0.15
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X_sat, y_sat) with extreme feature values that saturate float32 sigmoid.

    Scaling standard normals by 20 drives logistic decision_function beyond ±14,
    where sigmoid(14) = 1.0 exactly in float32 — the case that broke the old logit path.
    """
    rng = np.random.default_rng(seed)
    X_sat = (rng.standard_normal((n, n_features)) * 20).astype(np.float32)
    X_sat[rng.random((n, n_features)) < nan_frac] = np.nan
    y_sat = (np.nan_to_num(X_sat[:, 0]) > 0).astype(int)
    return X_sat, y_sat


@pytest.mark.skipif(not ONNX_AVAILABLE, reason="skl2onnx/onnxruntime not installed")
def test_onnx_parity_logreg():
    """LogReg + MedianImputeWithIndicator + Platt: parity ≤1e-5, ≥500 rows 15% NaN.

    Uses decision_function (raw margin) for the Platt input to avoid float32
    saturation: sigmoid(14.0) rounds to 1.0 in float32, so logit(1.0 - ε) ≠ 14.0.
    Validation set includes 75 saturated rows (features scaled ×20) to exercise
    the exact regime that caused the 1.48e-4 failure before this fix.
    """
    from market_anomaly_detector.selection import check_onnx_gate, make_logreg_pipeline

    X, y = _make_synthetic_data(500, 10, nan_frac=0.15)
    n_features = X.shape[1]
    X_train, y_train = X[:400], y[:400]

    # Parity check: all 500 normal rows + 75 saturated rows (575 total, ≥500)
    X_sat, y_sat = _make_saturated_rows(75, n_features, seed=99)
    X_check = np.concatenate([X, X_sat])
    y_check = np.concatenate([y, y_sat])

    pipe = make_logreg_pipeline()
    pipe.fit(X_train, y_train)

    val_raw = pipe.decision_function(X_check)
    platt = PlattCalibrator(use_logit=False).fit(val_raw, y_check)

    gate = check_onnx_gate("logreg", pipe, platt, X_check, n_features)

    assert gate.passed, f"LogReg ONNX gate failed: {gate.error}"
    assert gate.max_diff is not None
    assert gate.max_diff <= 1e-5, f"Max diff {gate.max_diff:.2e} exceeds 1e-5"


@pytest.mark.skipif(not ONNX_AVAILABLE, reason="skl2onnx/onnxruntime not installed")
def test_onnx_parity_hgb():
    """HGB + Platt exports to ONNX and matches sklearn within 1e-5, ≥500 rows 15% NaN.

    Uses decision_function (raw margin) for the Platt input (same fix as logreg).
    The HGB shim in calibration.py casts nodes_missing_value_tracks_true to int,
    fixing the skl2onnx 1.20 / onnx 1.23 incompatibility.
    Validation set includes 75 saturated rows to exercise float32-saturation regime.
    """
    from market_anomaly_detector.selection import check_onnx_gate, make_hgb_pipeline

    X, y = _make_synthetic_data(500, 10, nan_frac=0.15)
    n_features = X.shape[1]
    X_train, y_train = X[:400], y[:400]

    # Parity check: all 500 normal rows + 75 saturated rows (575 total, ≥500)
    X_sat, y_sat = _make_saturated_rows(75, n_features, seed=99)
    X_check = np.concatenate([X, X_sat])
    y_check = np.concatenate([y, y_sat])

    pipe = make_hgb_pipeline()
    pipe.fit(X_train, y_train)

    val_raw = pipe.decision_function(X_check)
    platt = PlattCalibrator(use_logit=False).fit(val_raw, y_check)

    gate = check_onnx_gate("hgb", pipe, platt, X_check, n_features)

    assert gate.passed, f"HGB ONNX gate failed: {gate.error}"
    assert gate.max_diff is not None
    assert gate.max_diff <= 1e-5, f"Max diff {gate.max_diff:.2e} exceeds 1e-5"


@pytest.mark.skipif(not ONNX_AVAILABLE, reason="skl2onnx/onnxruntime not installed")
def test_onnx_parity_twostage():
    """Two-stage + Platt exports to ONNX and matches sklearn within 1e-5, 500 rows 15% NaN.

    MedianImputeWithIndicator handles NaN on fit; IsoForestAppender uses OnnxSubEstimator.
    """
    from market_anomaly_detector.selection import check_onnx_gate, make_twostage_pipeline

    X, y = _make_synthetic_data(500, 10, nan_frac=0.15)
    n_features = X.shape[1]

    X_train, X_val = X[:400], X[400:]
    y_train, y_val = y[:400], y[400:]

    pipe = make_twostage_pipeline()
    pipe.fit(X_train, y_train)

    val_raw = pipe.predict_proba(X_val)[:, 1]
    platt = PlattCalibrator().fit(val_raw, y_val)

    gate = check_onnx_gate("twostage", pipe, platt, X_val, n_features)

    assert gate.passed, f"Two-stage ONNX gate failed: {gate.error}"
    assert gate.max_diff is not None
    assert gate.max_diff <= 1e-5, f"Max diff {gate.max_diff:.2e} exceeds 1e-5"


def test_median_impute_with_indicator_medians_from_fit_rows():
    """MedianImputeWithIndicator medians come from fit rows only, not transform rows."""
    imp = MedianImputeWithIndicator()
    # col0: values 1 and 3 (NaN excluded) → median 2.0
    # col1: values 4 and 6 (NaN excluded) → median 5.0
    X_fit = np.array([[1.0, np.nan], [3.0, 4.0], [np.nan, 6.0]])
    imp.fit(X_fit)
    assert imp.medians_[0] == pytest.approx(2.0)
    assert imp.medians_[1] == pytest.approx(5.0)

    # Transform on different data: NaN slots filled with fit medians
    X_test = np.array([[np.nan, np.nan], [10.0, 20.0]])
    out = imp.transform(X_test)
    assert out.shape == (2, 4)  # 2 features + 2 indicators
    assert out[0, 0] == pytest.approx(2.0)  # col0 NaN → filled with fit median
    assert out[0, 1] == pytest.approx(5.0)  # col1 NaN → filled with fit median
    assert out[0, 2] == pytest.approx(1.0)  # was-missing indicator col0
    assert out[0, 3] == pytest.approx(1.0)  # was-missing indicator col1
    assert out[1, 0] == pytest.approx(10.0)  # not NaN → unchanged
    assert out[1, 2] == pytest.approx(0.0)  # not missing → indicator = 0


# ── Core-feature mask ─────────────────────────────────────────────────────────


def test_core_feature_mask_uses_spec_v1_col_names():
    """_core_feature_mask uses SPEC_V1 instrument_id, not name prefixes.

    sp500_lr_1w (instrument_id='msci_usa') and spread_10y_3m
    (instrument_id='us_10y_yield') are both core; a NaN in either must
    mark that row as excluded (mask=False).
    """
    from market_anomaly_detector.selection import _core_feature_mask

    # Two rows, three columns: one vix col, one sp500 col, one spread col.
    # Row 0: all present → keep.
    # Row 1: sp500_lr_1w is NaN → exclude.
    # Row 2: spread_10y_3m is NaN → exclude.
    cols = ["vix_lr_1w", "sp500_lr_1w", "spread_10y_3m"]
    X_arr = np.array(
        [
            [15.0, 0.01, 0.5],  # row 0: all present
            [15.0, np.nan, 0.5],  # row 1: sp500 NaN (msci_usa core)
            [15.0, 0.01, np.nan],  # row 2: spread_10y NaN (us_10y_yield core)
        ]
    )
    X_df = pd.DataFrame(X_arr, columns=cols)
    mask = _core_feature_mask(X_df, X_arr)
    assert mask[0] is np.bool_(True), "Row 0 (all present) should be kept"
    assert mask[1] is np.bool_(False), "Row 1 (sp500_lr_1w NaN) should be excluded"
    assert mask[2] is np.bool_(False), "Row 2 (spread_10y_3m NaN) should be excluded"


# ── Calibration diagnostic ────────────────────────────────────────────────────


def test_collect_calibration_diagnostics_shape_and_contract():
    """collect_calibration_diagnostics returns one row per fold×candidate with valid values.

    Uses a minimal synthetic dataset (no DB) over a 2-fold span.  Checks:
    - one row per fold
    - all probability columns in [0, 1]
    - val_calm_far in [0, 1]
    - Platt a and b are finite
    - threshold is finite (or inf when no calm rows)
    """
    from market_anomaly_detector.eval_harness import FoldSpec
    from market_anomaly_detector.selection import (
        collect_calibration_diagnostics,
        make_logreg_pipeline,
    )

    rng = np.random.default_rng(77)
    n_features = 8

    # Build synthetic labels + feature data spanning 2010–2011 (2 test folds)
    # with enough positives for make_folds_v2 requirements.
    # Build manually: 4 years of train/val history + 2 test years
    years = list(range(2006, 2012))  # 6 years
    all_weeks: list[date] = []
    for yr in years:
        t = date(yr, 1, 3)
        while t.year == yr:
            if t.weekday() == 1:
                all_weeks.append(t)
            t = t + timedelta(days=1)

    n_weeks = len(all_weeks)
    detect = np.zeros(n_weeks, dtype=int)
    # Both train (2006–2007) and later windows need both classes for the pipeline.
    for i, w in enumerate(all_weeks):
        if 2006 <= w.year <= 2007:
            detect[i] = int(rng.random() < 0.15)
        elif w.year >= 2008:
            detect[i] = int(rng.random() < 0.30)

    labels_df = pd.DataFrame({"week": all_weeks, "value": detect})
    X_arr = rng.standard_normal((n_weeks, n_features)).astype(float)
    X_arr[rng.random((n_weeks, n_features)) < 0.10] = np.nan
    X_df = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(n_features)])
    ku_mask = np.zeros(n_weeks, dtype=bool)

    # Build two folds manually (2010 and 2011 as test years)
    def _lo(yr: int) -> int:
        return next(i for i, w in enumerate(all_weeks) if w.year == yr)

    def _hi(yr: int) -> int:
        idxs = [i for i, w in enumerate(all_weeks) if w.year == yr]
        return idxs[-1] + 1

    folds = [
        FoldSpec(
            test_year=2010,
            val_lo=_lo(2008),
            test_lo=_lo(2010),
            test_hi=_hi(2010),
            val_start=all_weeks[_lo(2008)],
            val_end=all_weeks[_lo(2010) - 1],
            test_start=all_weeks[_lo(2010)],
            test_end=all_weeks[_hi(2010) - 1],
        ),
        FoldSpec(
            test_year=2011,
            val_lo=_lo(2008),
            test_lo=_lo(2011),
            test_hi=_hi(2011),
            val_start=all_weeks[_lo(2008)],
            val_end=all_weeks[_lo(2011) - 1],
            test_start=all_weeks[_lo(2011)],
            test_end=all_weeks[_hi(2011) - 1],
        ),
    ]

    rows = collect_calibration_diagnostics(
        "logreg", make_logreg_pipeline, X_df, labels_df, folds, ku_mask
    )

    assert len(rows) == 2, f"Expected 2 rows (one per fold), got {len(rows)}"
    fold_years = {r["fold_year"] for r in rows}
    assert fold_years == {2010, 2011}

    for r in rows:
        assert r["candidate"] == "logreg"
        # val_pos <= val_total
        assert 0 <= r["val_pos"] <= r["val_total"]
        # Platt params finite
        assert math.isfinite(r["platt_a"]) and math.isfinite(r["platt_b"])
        # Calibrated test probabilities in [0,1]
        assert 0.0 <= r["cal_test_min"] <= r["cal_test_med"] <= r["cal_test_max"] <= 1.0
        # val calm FAR in [0,1] or nan (no calm rows)
        assert math.isnan(r["val_calm_far"]) or 0.0 <= r["val_calm_far"] <= 1.0


# ── Integration test (requires MAD_TEST_DATABASE_URL) ────────────────────────


@pytest.mark.db
def test_run_selection_smoke(db_session, tmp_path):
    """Integration smoke-test: synthetic price + label rows → run_selection.

    Inserts minimal synthetic data for all SPEC_V1 instruments and a short
    label set, then calls run_selection(n_boot=20) and verifies the report
    is written with valid VIX level event F1 and all three candidates listed.
    """
    from datetime import date, timedelta

    from market_anomaly_detector.db.models import InstrumentRow, LabelRow, PriceRow, SymbolMapRow
    from market_anomaly_detector.features.spec import SPEC_V1
    from market_anomaly_detector.selection import run_selection

    # Build the set of instrument_ids needed
    inst_ids = set()
    for s in SPEC_V1:
        inst_ids.add(s.instrument_id)
        if s.instrument_id_2:
            inst_ids.add(s.instrument_id_2)

    # Insert instrument rows
    for iid in inst_ids:
        db_session.add(
            InstrumentRow(
                id=iid,
                name=iid,
                asset_class="index",
                units="index_points",
                quote="USD",
                native_frequency="daily",
            )
        )
    db_session.flush()

    # Insert symbol_map so load_prices can find the provider
    for iid in inst_ids:
        db_session.add(
            SymbolMapRow(
                instrument_id=iid,
                provider="yahoo",
                symbol=f"{iid}=X",
                priority=1,
                fidelity="eod",
            )
        )
    db_session.flush()

    # Generate daily price rows (2000-01-03 to 2017-12-29, Mon-Fri)
    rng = np.random.default_rng(7)
    start_date = date(2000, 1, 3)
    end_date = date(2017, 12, 29)
    price_rows = []
    cur = start_date
    while cur <= end_date:
        if cur.weekday() < 5:  # Mon-Fri
            for iid in inst_ids:
                # VIX: mean ~20, log-normal; others: random walk
                base = 20.0 if iid == "vix" else 100.0
                val = float(base + rng.normal(0, 1))
                val = max(val, 0.5)
                price_rows.append(
                    PriceRow(
                        instrument_id=iid,
                        provider="yahoo",
                        obs_date=cur,
                        value=val,
                        known_at=cur,
                        ingested_at=pd.Timestamp.now(tz="UTC").to_pydatetime(),
                    )
                )
        cur += timedelta(days=1)
    db_session.bulk_save_objects(price_rows)
    db_session.flush()

    # Generate weekly label rows (Tuesdays 2004-2017).
    # The 2008 fold's train is 2004-2005 and its val is 2006-2007;
    # make_folds_v2 requires min_val_positives=8 and classifiers need both
    # classes in train, so anomalies must appear in all these windows.
    # 2004-2005: ~15% (train coverage).
    # 2006-2007: ~25% (val window for the 2008 fold).
    # 2008-2009, 2011-2012: ~50% (high-stress periods).
    label_set = "test_ews"
    tues = date(2004, 1, 6)
    label_rows = []
    while tues <= date(2017, 12, 26):
        yr = tues.year
        if 2004 <= yr <= 2005:
            val = 1 if rng.random() < 0.15 else 0
        elif 2006 <= yr <= 2007:
            val = 1 if rng.random() < 0.25 else 0
        elif (2008 <= yr <= 2009) or (2011 <= yr <= 2012):
            val = 1 if rng.random() < 0.5 else 0
        else:
            val = 0
        label_rows.append(LabelRow(label_set=label_set, week=tues, value=val, source="test"))
        tues += timedelta(weeks=1)
    db_session.bulk_save_objects(label_rows)
    db_session.flush()

    # Run selection (small n_boot for speed; 2008-2010 only to keep it fast)
    out_dir = tmp_path / "reports"
    out_dir.mkdir()
    rc = run_selection(
        db_session,
        out_dir,
        label_set=label_set,
        first_test_year=2008,
        last_test_year=2010,
        n_boot=20,
        csv_path=None,
        skip_bootstrap=False,
    )
    assert rc == 0, "run_selection returned non-zero exit code"

    report_files = list(out_dir.glob("*-model-selection.md"))
    assert report_files, "No model-selection report was written"
    report = report_files[0].read_text(encoding="utf-8")

    # Report must mention all three candidates
    for cname in ("logreg", "hgb", "twostage"):
        assert cname in report, f"Candidate '{cname}' not found in report"

    # VIX level event F1 must be present and not NaN
    assert "vix_level" in report, "vix_level not in report"
    assert "not evaluated" not in report.split("vix_level")[1][:50], (
        "vix_level shows 'not evaluated'"
    )

    # Bootstrap section must show actual n_boot (20), not hardcoded 2000
    assert "20 resamples" in report, "Bootstrap heading does not show actual n_boot"


# ── export-dev-predictions ────────────────────────────────────────────────────


def test_export_alerts_match_select():
    """build_export_df alert columns must exactly match evaluate_candidate fold results.

    Calls evaluate_candidate("hgb") and _evaluate_vix_baselines() on synthetic
    data, then calls build_export_df() with the same inputs and verifies that
    the alert sets in the CSV are identical to what the FoldResults contain —
    i.e., the CSV assembly code is consistent with the selection code path.
    """
    from market_anomaly_detector.episodes import extract_episodes
    from market_anomaly_detector.eval_harness import FoldSpec
    from market_anomaly_detector.selection import (
        CANDIDATES,
        _evaluate_vix_baselines,
        build_export_df,
        evaluate_candidate,
    )

    rng = np.random.default_rng(55)
    n_features = 8

    years = list(range(2006, 2012))
    all_weeks: list[date] = []
    for yr in years:
        t = date(yr, 1, 3)
        while t.year == yr:
            if t.weekday() == 1:
                all_weeks.append(t)
            t += timedelta(days=1)

    n_weeks = len(all_weeks)
    detect = np.zeros(n_weeks, dtype=int)
    for i, w in enumerate(all_weeks):
        if 2006 <= w.year <= 2007:
            detect[i] = int(rng.random() < 0.15)
        elif w.year >= 2008:
            detect[i] = int(rng.random() < 0.30)

    labels_df = pd.DataFrame({"week": all_weeks, "value": detect})
    X_arr = rng.standard_normal((n_weeks, n_features)).astype(float)
    X_arr[rng.random((n_weeks, n_features)) < 0.10] = np.nan
    X_df = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(n_features)])
    ku_mask = np.zeros(n_weeks, dtype=bool)
    vix_close = rng.uniform(10.0, 40.0, n_weeks)

    def _lo(yr: int) -> int:
        return next(i for i, w in enumerate(all_weeks) if w.year == yr)

    def _hi(yr: int) -> int:
        idxs = [i for i, w in enumerate(all_weeks) if w.year == yr]
        return idxs[-1] + 1

    folds = [
        FoldSpec(
            test_year=2010,
            val_lo=_lo(2008),
            test_lo=_lo(2010),
            test_hi=_hi(2010),
            val_start=all_weeks[_lo(2008)],
            val_end=all_weeks[_lo(2010) - 1],
            test_start=all_weeks[_lo(2010)],
            test_end=all_weeks[_hi(2010) - 1],
        ),
        FoldSpec(
            test_year=2011,
            val_lo=_lo(2008),
            test_lo=_lo(2011),
            test_hi=_hi(2011),
            val_start=all_weeks[_lo(2008)],
            val_end=all_weeks[_lo(2011) - 1],
            test_start=all_weeks[_lo(2011)],
            test_end=all_weeks[_hi(2011) - 1],
        ),
    ]

    dev_start, dev_end = date(2010, 1, 1), date(2011, 12, 31)
    episodes = [
        e for e in extract_episodes(labels_df, merge_gap=2) if dev_start <= e.start <= dev_end
    ]

    hgb_result = evaluate_candidate("hgb", CANDIDATES["hgb"], X_df, labels_df, folds, ku_mask)
    vix_baselines = _evaluate_vix_baselines(labels_df, folds, vix_close, ku_mask)
    vix_result = vix_baselines["vix_level"]

    df = build_export_df(hgb_result, vix_result, all_weeks, vix_close, ku_mask, episodes)

    # ── schema checks ─────────────────────────────────────────────────────────
    expected_cols = {
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
    }
    assert expected_cols.issubset(set(df.columns)), (
        f"Missing columns: {expected_cols - set(df.columns)}"
    )
    assert df["week"].is_unique, "Duplicate weeks in export"
    assert all(dev_start <= w <= dev_end for w in df["week"]), "Week outside dev window"

    # ── alert parity: hgb detect ──────────────────────────────────────────────
    export_hgb = set(df[df["hgb_alert"] == 1]["week"])
    fold_hgb = {w for fr in hgb_result.detect_folds for w, a in zip(fr.weeks, fr.alerts) if a == 1}
    assert export_hgb == fold_hgb, f"HGB detect alert mismatch: {export_hgb ^ fold_hgb}"

    # ── alert parity: vix_level detect ────────────────────────────────────────
    export_vix = set(df[df["vix_level_alert"] == 1]["week"])
    fold_vix = {w for fr in vix_result.detect_folds for w, a in zip(fr.weeks, fr.alerts) if a == 1}
    assert export_vix == fold_vix, f"VIX detect alert mismatch: {export_vix ^ fold_vix}"

    # ── alert parity: hgb predict ─────────────────────────────────────────────
    export_hgb_pred = set(df[df["hgb_predict_alert"] == 1]["week"])
    fold_hgb_pred = {
        w for fr in hgb_result.predict_folds for w, a in zip(fr.weeks, fr.alerts) if a == 1
    }
    assert export_hgb_pred == fold_hgb_pred, (
        f"HGB predict alert mismatch: {export_hgb_pred ^ fold_hgb_pred}"
    )

    # ── alert parity: vix predict ─────────────────────────────────────────────
    export_vix_pred = set(df[df["vix_predict_alert"] == 1]["week"])
    fold_vix_pred = {
        w for fr in vix_result.predict_folds for w, a in zip(fr.weeks, fr.alerts) if a == 1
    }
    assert export_vix_pred == fold_vix_pred, (
        f"VIX predict alert mismatch: {export_vix_pred ^ fold_vix_pred}"
    )

    # ── thresholds and probabilities ──────────────────────────────────────────
    # Rows with hgb_alert set must have a valid (non-NaN) hgb_prob and threshold
    hgb_rows = df[df["hgb_alert"].notna()]
    assert hgb_rows["hgb_prob"].notna().all(), "Some HGB rows have NaN prob but non-null alert"
    assert hgb_rows["hgb_threshold"].notna().all(), "Some HGB rows have NaN threshold"
    assert (hgb_rows["hgb_prob"].between(0.0, 1.0)).all(), "HGB probs outside [0, 1]"


# ── predict-head boundary purge ───────────────────────────────────────────────


def test_predict_head_no_boundary_leakage():
    """After purging, no predict-head training row's target window overlaps val,
    and no val row's target window overlaps test.

    predict[t] = any(labels[t+1..t+h]); a row at t is leaky when t+h >= boundary.
    """
    from market_anomaly_detector.eval_harness import make_folds_v2
    from market_anomaly_detector.labels import build_targets
    from market_anomaly_detector.selection import HORIZON

    rng = np.random.default_rng(42)
    start = date(2000, 1, 4)
    weeks = [start + timedelta(weeks=i) for i in range(1000)]
    detect = (rng.uniform(size=1000) > 0.78).astype(int)
    labels_df = pd.DataFrame({"week": weeks, "value": detect})

    folds = make_folds_v2(weeks, detect, first_test_year=2008, last_test_year=2017)
    predict_arr = build_targets(labels_df, horizon=HORIZON)["predict"].values

    for fold in folds:
        # Mirror the purge logic from evaluate_candidate.
        train_pred_mask = ~np.isnan(predict_arr[: fold.val_lo])
        train_purge_from = max(0, fold.val_lo - HORIZON)
        train_pred_mask[train_purge_from:] = False

        retained = np.flatnonzero(train_pred_mask)
        assert (retained + HORIZON < fold.val_lo).all(), (
            f"Fold {fold.test_year}: leaky training predict index detected"
        )

        val_pred_mask = ~np.isnan(predict_arr[fold.val_lo : fold.test_lo])
        val_purge_from = max(0, fold.test_lo - HORIZON - fold.val_lo)
        val_pred_mask[val_purge_from:] = False

        retained_val = fold.val_lo + np.flatnonzero(val_pred_mask)
        assert (retained_val + HORIZON < fold.test_lo).all(), (
            f"Fold {fold.test_year}: leaky validation predict index detected"
        )
