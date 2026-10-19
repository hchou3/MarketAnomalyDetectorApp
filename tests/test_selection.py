"""Tests for Phase 2 Step 2.4 — candidate model selection.

All tests use synthetic data; no DB or network required.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pytest

from market_anomaly_detector.calibration import PlattCalibrator
from market_anomaly_detector.episodes import Episode
from market_anomaly_detector.eval_harness import choose_threshold
from market_anomaly_detector.selection import (
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


def _make_synthetic_data(n: int = 200, n_features: int = 10, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    y = (X[:, 0] + rng.standard_normal(n) * 0.5 > 0).astype(int)
    return X, y


try:
    import onnxruntime  # noqa: F401
    import skl2onnx  # noqa: F401

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


@pytest.mark.skipif(not ONNX_AVAILABLE, reason="skl2onnx/onnxruntime not installed")
def test_onnx_parity_logreg():
    """LogReg pipeline + Platt exports to ONNX and matches sklearn within 1e-5."""
    from market_anomaly_detector.calibration import onnx_export_with_platt
    from market_anomaly_detector.selection import make_logreg_pipeline

    X, y = _make_synthetic_data(200, 10)
    n_features = X.shape[1]

    # Split 150 train / 50 val
    X_train, X_val = X[:150], X[150:]
    y_train, y_val = y[:150], y[150:]

    pipe = make_logreg_pipeline()
    pipe.fit(X_train, y_train)

    val_raw = pipe.predict_proba(X_val)[:, 1]
    platt = PlattCalibrator().fit(val_raw, y_val)

    onnx_bytes = onnx_export_with_platt(pipe, platt, n_features)

    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_bytes)
    input_name = sess.get_inputs()[0].name
    ort_probs = sess.run(["calibrated_probability"], {input_name: X_val})[0].flatten()

    sklearn_raw = pipe.predict_proba(X_val)[:, 1]
    sklearn_probs = platt.transform(sklearn_raw).astype(np.float32)

    max_diff = float(np.max(np.abs(ort_probs - sklearn_probs)))
    assert max_diff <= 1e-5, f"Max diff {max_diff:.2e} exceeds 1e-5"


@pytest.mark.skipif(not ONNX_AVAILABLE, reason="skl2onnx/onnxruntime not installed")
def test_onnx_parity_hgb():
    """HGB gate behavior: either parity ≤1e-5 (pass) or clean failure (gate=False).

    skl2onnx 1.20 + sklearn 1.6.1 has a known bug where HGB tree node IDs are stored
    as Python booleans, causing conversion to fail. The gate must report this cleanly
    (passed=False, no crash) so HGB is excluded from selection rather than raising.
    """
    from market_anomaly_detector.selection import check_onnx_gate, make_hgb_pipeline

    X, y = _make_synthetic_data(200, 10)
    n_features = X.shape[1]

    X_train, X_val = X[:150], X[150:]
    y_train, y_val = y[:150], y[150:]

    pipe = make_hgb_pipeline()
    pipe.fit(X_train, y_train)

    val_raw = pipe.predict_proba(X_val)[:, 1]
    platt = PlattCalibrator().fit(val_raw, y_val)

    gate = check_onnx_gate("hgb", pipe, platt, X_val, n_features)

    if gate.passed:
        # If export works (future library fix), verify parity
        assert gate.max_diff is not None
        assert gate.max_diff <= 1e-5, f"Max diff {gate.max_diff:.2e} exceeds 1e-5"
    else:
        # Known skl2onnx 1.20/sklearn 1.6.1 incompatibility — gate must report cleanly
        assert gate.error is not None, "Gate should report an error message on failure"
        assert gate.max_diff is None
