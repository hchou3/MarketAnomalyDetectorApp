"""Tests for Phase 2 Step 2.4 — candidate model selection.

All tests use synthetic data; no DB or network required.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

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


@pytest.mark.skipif(not ONNX_AVAILABLE, reason="skl2onnx/onnxruntime not installed")
def test_onnx_parity_logreg():
    """LogReg + MedianImputeWithIndicator + Platt: parity ≤1e-5, 500 rows 15% NaN."""
    from market_anomaly_detector.selection import check_onnx_gate, make_logreg_pipeline

    X, y = _make_synthetic_data(500, 10, nan_frac=0.15)
    n_features = X.shape[1]

    X_train, X_val = X[:400], X[400:]
    y_train, y_val = y[:400], y[400:]

    pipe = make_logreg_pipeline()
    pipe.fit(X_train, y_train)

    val_raw = pipe.predict_proba(X_val)[:, 1]
    platt = PlattCalibrator().fit(val_raw, y_val)

    gate = check_onnx_gate("logreg", pipe, platt, X_val, n_features)

    assert gate.passed, f"LogReg ONNX gate failed: {gate.error}"
    assert gate.max_diff is not None
    assert gate.max_diff <= 1e-5, f"Max diff {gate.max_diff:.2e} exceeds 1e-5"


@pytest.mark.skipif(not ONNX_AVAILABLE, reason="skl2onnx/onnxruntime not installed")
def test_onnx_parity_hgb():
    """HGB + Platt exports to ONNX and matches sklearn within 1e-5, 500 rows 15% NaN.

    The HGB shim in calibration.py casts nodes_missing_value_tracks_true to int,
    fixing the skl2onnx 1.20 / onnx 1.23 incompatibility.
    """
    from market_anomaly_detector.selection import check_onnx_gate, make_hgb_pipeline

    X, y = _make_synthetic_data(500, 10, nan_frac=0.15)
    n_features = X.shape[1]

    X_train, X_val = X[:400], X[400:]
    y_train, y_val = y[:400], y[400:]

    pipe = make_hgb_pipeline()
    pipe.fit(X_train, y_train)

    val_raw = pipe.predict_proba(X_val)[:, 1]
    platt = PlattCalibrator().fit(val_raw, y_val)

    gate = check_onnx_gate("hgb", pipe, platt, X_val, n_features)

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
