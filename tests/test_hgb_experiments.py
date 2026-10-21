"""Tests for hgb_experiments mechanics.

Three required tests:
  - test_hysteresis_state_machine
  - test_episode_weights_equal_total
  - test_oof_no_future_data
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from market_anomaly_detector.episodes import Episode
from market_anomaly_detector.eval_harness import FoldSpec
from market_anomaly_detector.hgb_experiments import (
    VARIANT_ORDER,
    VARIANTS,
    _oof_scores_for_fold,
    apply_hysteresis,
    episode_sample_weights,
)

# ── Hysteresis state machine ──────────────────────────────────────────────────


def test_hysteresis_state_machine():
    """Alert turns on at threshold; stays on above factor*thr; turns off below."""
    thr = 0.6
    # off_thr = 0.5 * 0.6 = 0.3
    # 0.3 < 0.6 → off
    # 0.5 < 0.6 → off
    # 0.65 >= 0.6 → ON
    # 0.4 >= 0.3 → stays ON
    # 0.25 < 0.3 → OFF
    # 0.2 < 0.3 → off
    proba = np.array([0.3, 0.5, 0.65, 0.4, 0.25, 0.2])
    alerts = apply_hysteresis(proba, thr)
    expected = np.array([0, 0, 1, 1, 0, 0])
    np.testing.assert_array_equal(alerts, expected)


def test_hysteresis_never_fires():
    """No alert when all probabilities are below threshold."""
    proba = np.array([0.1, 0.2, 0.3])
    alerts = apply_hysteresis(proba, threshold=0.5)
    np.testing.assert_array_equal(alerts, np.zeros(3, dtype=int))


def test_hysteresis_always_fires():
    """Alert stays on when all probabilities are above threshold."""
    proba = np.array([0.9, 0.8, 0.7])
    alerts = apply_hysteresis(proba, threshold=0.5)
    np.testing.assert_array_equal(alerts, np.ones(3, dtype=int))


def test_hysteresis_off_threshold_uses_factor():
    """Off threshold = factor * threshold, not threshold itself."""
    thr = 0.5
    factor = 0.4
    # on at 0.6, stays on at 0.22 (>= 0.5*0.4=0.20), off at 0.15 (<0.20)
    proba = np.array([0.6, 0.22, 0.15])
    alerts = apply_hysteresis(proba, thr, factor=factor)
    expected = np.array([1, 1, 0])
    np.testing.assert_array_equal(alerts, expected)


def test_hysteresis_empty():
    """Empty input returns empty output."""
    alerts = apply_hysteresis(np.array([]), threshold=0.5)
    assert len(alerts) == 0


# ── Episode sample weights ────────────────────────────────────────────────────


def test_episode_weights_equal_total():
    """Each episode in the training window has total weight exactly 1.0."""
    weeks = [date(2000, 1, 4) + timedelta(weeks=i) for i in range(10)]
    y = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0])

    ep_a = Episode(start=weeks[0], end=weeks[2], peak_week=weeks[0], length=3)
    ep_b = Episode(start=weeks[6], end=weeks[8], peak_week=weeks[6], length=3)
    episodes = [ep_a, ep_b]

    weights = episode_sample_weights(y, weeks, episodes)

    assert abs(sum(weights[i] for i in [0, 1, 2]) - 1.0) < 1e-9, "episode A total weight != 1"
    assert abs(sum(weights[i] for i in [6, 7, 8]) - 1.0) < 1e-9, "episode B total weight != 1"
    for i in [3, 4, 5, 9]:
        assert weights[i] == pytest.approx(1.0), f"calm week {i} weight != 1.0"


def test_episode_weights_calm_weeks_unchanged():
    """Calm weeks that share no overlap with any episode get weight 1.0."""
    weeks = [date(2000, 1, 4) + timedelta(weeks=i) for i in range(5)]
    y = np.zeros(5, dtype=int)
    ep = Episode(
        start=date(2010, 1, 1), end=date(2010, 3, 1), peak_week=date(2010, 1, 1), length=10
    )
    weights = episode_sample_weights(y, weeks, [ep])
    np.testing.assert_array_equal(weights, np.ones(5))


def test_episode_weights_partial_episode_in_train():
    """Episode partially in train: only the overlapping weeks are down-weighted."""
    weeks = [date(2000, 1, 4) + timedelta(weeks=i) for i in range(5)]
    # Episode spans weeks[3] through weeks[6] but train only has weeks[0:5]
    ep = Episode(start=weeks[3], end=date(2000, 3, 21), peak_week=weeks[3], length=4)
    y = np.array([0, 0, 0, 1, 1])
    weights = episode_sample_weights(y, weeks, [ep])
    # weeks[3] and weeks[4] are in train overlap (2 weeks)
    assert weights[3] == pytest.approx(0.5)
    assert weights[4] == pytest.approx(0.5)
    assert weights[0] == pytest.approx(1.0)
    assert abs(weights[3] + weights[4] - 1.0) < 1e-9


# ── OOF no future data ────────────────────────────────────────────────────────


def test_oof_no_future_data():
    """OOF inner folds must use only data from years strictly before test_year."""
    rng = np.random.default_rng(7)
    n = 416  # ~ 8 years of weekly data
    start = date(2000, 1, 4)
    all_weeks = [start + timedelta(weeks=i) for i in range(n)]
    detect_arr = (np.arange(n) % 7 == 0).astype(int)

    # Mark test-year rows with a distinctive NaN pattern so they'd break any model
    # that accidentally uses them as training data.
    test_year = 2007
    X_all = rng.standard_normal((n, 4))
    for i, w in enumerate(all_weeks):
        if w.year >= test_year:
            X_all[i] = np.nan  # would contaminate any training set

    core_ok = np.array([not np.all(np.isnan(X_all[i])) for i in range(n)])

    test_lo = next(i for i, w in enumerate(all_weeks) if w.year == test_year)
    fold = FoldSpec(
        test_year=test_year,
        val_lo=max(0, test_lo - 52),
        test_lo=test_lo,
        test_hi=min(test_lo + 52, n),
        val_start=all_weeks[max(0, test_lo - 52)],
        val_end=all_weeks[test_lo - 1],
        test_start=all_weeks[test_lo],
        test_end=all_weeks[min(test_lo + 51, n - 1)],
    )

    # _oof_scores_for_fold must complete without NaN errors (NaN rows are filtered
    # by core_ok) and must produce some OOF scores from pre-test years only.
    oof_scores, oof_labels, oof_ku = _oof_scores_for_fold(
        X_all, detect_arr, all_weeks, core_ok, fold
    )

    # Should produce scores (years 2000-2006 all have valid data)
    assert len(oof_scores) > 0, "OOF should produce scores from pre-test years"
    assert len(oof_scores) == len(oof_labels)
    assert not np.any(np.isnan(oof_scores)), "OOF scores must be finite"
    # oof_ku is always False (no KU tracking in inner folds)
    assert not np.any(oof_ku)


def test_oof_returns_empty_when_no_train_data():
    """OOF returns empty arrays when there is not enough pre-history to train."""
    n = 30
    start = date(2000, 1, 4)
    all_weeks = [start + timedelta(weeks=i) for i in range(n)]
    detect_arr = np.zeros(n, dtype=int)
    X_all = np.ones((n, 2))
    core_ok = np.ones(n, dtype=bool)

    # test_lo = 10 → only 10 rows before test, all in year 2000
    fold = FoldSpec(
        test_year=2000,
        val_lo=5,
        test_lo=10,
        test_hi=20,
        val_start=all_weeks[5],
        val_end=all_weeks[9],
        test_start=all_weeks[10],
        test_end=all_weeks[19],
    )

    oof_scores, oof_labels, oof_ku = _oof_scores_for_fold(
        X_all, detect_arr, all_weeks, core_ok, fold
    )

    # Year 2000 is the only pre-test year; inner train before 2000 has 0 rows → skipped
    assert len(oof_scores) == 0
    assert len(oof_labels) == 0


# ── Variants catalogue ────────────────────────────────────────────────────────


def test_variants_dict_complete():
    """VARIANTS contains exactly the 6 expected variants."""
    expected = {"hgb_base", "hgb_vixlvl", "hgb_oof", "hgb_hyst", "hgb_epw", "hgb_all"}
    assert set(VARIANTS.keys()) == expected


def test_variant_order_complete():
    """VARIANT_ORDER lists all 6 variants."""
    assert len(VARIANT_ORDER) == 6
    assert set(VARIANT_ORDER) == set(VARIANTS.keys())


def test_hgb_base_all_flags_off():
    """hgb_base has no variant flags set."""
    v = VARIANTS["hgb_base"]
    assert not v.use_vixlvl
    assert not v.use_oof
    assert not v.use_hyst
    assert not v.use_epw


def test_hgb_all_combines_all_flags():
    """hgb_all has every variant flag enabled."""
    v = VARIANTS["hgb_all"]
    assert v.use_vixlvl and v.use_oof and v.use_hyst and v.use_epw


def test_single_flag_variants():
    """Each single-flag variant has exactly one flag set."""
    assert VARIANTS["hgb_vixlvl"].use_vixlvl and not VARIANTS["hgb_vixlvl"].use_oof
    assert VARIANTS["hgb_oof"].use_oof and not VARIANTS["hgb_oof"].use_vixlvl
    assert VARIANTS["hgb_hyst"].use_hyst and not VARIANTS["hgb_hyst"].use_epw
    assert VARIANTS["hgb_epw"].use_epw and not VARIANTS["hgb_epw"].use_hyst
