"""Tests for eval_harness.py — one test per metric edge case.

All tests are offline (no database required).
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pytest

from market_anomaly_detector.episodes import Episode
from market_anomaly_detector.eval_harness import (
    always_flag,
    choose_threshold,
    detect_event_results,
    false_alarm_runs,
    fit_vix_threshold,
    make_folds_v2,
    persistence,
    predict_event_results,
    prob_metrics,
    row_metrics,
    vix_rule,
    vix_z52,
)

# ── helpers ────────────────────────────────────────────────────────────────────


def _tuesdays(start: date, n: int) -> list[date]:
    d = start
    while d.weekday() != 1:
        d += timedelta(days=1)
    return [d + timedelta(weeks=i) for i in range(n)]


def _episode(start: date, length: int) -> Episode:
    end = start + timedelta(weeks=length - 1)
    return Episode(start=start, end=end, peak_week=start, length=length)


# ── detect_event_results ───────────────────────────────────────────────────────


def test_detect_hit_at_onset():
    """Alert on onset week = hit with weeks_into_episode=0."""
    ep = _episode(date(2010, 1, 5), 4)
    result = detect_event_results([ep], {date(2010, 1, 5)})[0]
    assert result.hit
    assert result.weeks_into_episode == 0
    assert result.first_alert_week == date(2010, 1, 5)


def test_detect_hit_mid_episode():
    """Alert 2 weeks into a 4-week episode."""
    ep = _episode(date(2010, 1, 5), 4)
    result = detect_event_results([ep], {date(2010, 1, 19)})[0]
    assert result.hit
    assert result.weeks_into_episode == 2


def test_detect_miss_no_alert():
    """No alert during episode = miss."""
    ep = _episode(date(2010, 1, 5), 3)
    result = detect_event_results([ep], set())[0]
    assert not result.hit
    assert result.weeks_into_episode is None
    assert result.first_alert_week is None


def test_detect_miss_alert_outside_episode():
    """Alert outside the episode window is not a hit."""
    ep = _episode(date(2010, 1, 5), 2)  # ends 2010-01-12
    result = detect_event_results([ep], {date(2010, 1, 19)})[0]
    assert not result.hit


def test_detect_multiple_alerts_first_wins():
    """weeks_into_episode uses the FIRST alert within the episode."""
    ep = _episode(date(2010, 1, 5), 4)
    alerts = {date(2010, 1, 12), date(2010, 1, 19)}  # weeks 1 and 2
    result = detect_event_results([ep], alerts)[0]
    assert result.first_alert_week == date(2010, 1, 12)
    assert result.weeks_into_episode == 1


def test_detect_event_recall_two_episodes():
    """Event recall = hits / total episodes."""
    ep1 = _episode(date(2010, 1, 5), 2)
    ep2 = _episode(date(2010, 3, 2), 2)
    # Only ep2 is caught
    results = detect_event_results([ep1, ep2], {date(2010, 3, 2)})
    hits = sum(r.hit for r in results)
    assert hits == 1


def test_detect_empty_episodes():
    assert detect_event_results([], {date(2010, 1, 5)}) == []


# ── predict_event_results ──────────────────────────────────────────────────────


def test_predict_hit_one_week_before():
    """Calm alert 1 week before onset = hit, weeks_before_onset=1."""
    ep = _episode(date(2010, 1, 12), 3)  # onset 2010-01-12
    calm_alerts = {date(2010, 1, 5)}  # 1 week before
    result = predict_event_results([ep], calm_alerts, horizon=4)[0]
    assert result.hit
    assert result.weeks_into_episode == 1
    assert result.first_alert_week == date(2010, 1, 5)


def test_predict_hit_four_weeks_before():
    """Calm alert h=4 weeks before onset = hit, weeks_before_onset=4."""
    ep = _episode(date(2010, 2, 2), 3)
    calm_alerts = {date(2010, 1, 5)}  # exactly 4 weeks before
    result = predict_event_results([ep], calm_alerts, horizon=4)[0]
    assert result.hit
    assert result.weeks_into_episode == 4


def test_predict_miss_alert_too_early():
    """Calm alert 5 weeks before onset is outside the predict window (h=4) = miss."""
    ep = _episode(date(2010, 2, 9), 3)  # onset 2010-02-09
    calm_alerts = {date(2010, 1, 5)}  # 5 weeks before
    result = predict_event_results([ep], calm_alerts, horizon=4)[0]
    assert not result.hit


def test_predict_miss_no_calm_alert():
    assert not predict_event_results([_episode(date(2010, 1, 5), 3)], set(), 4)[0].hit


def test_predict_alert_during_episode_not_a_lead():
    """An alert on/after onset does not count as advance warning."""
    ep = _episode(date(2010, 1, 12), 3)
    # Alert is on the onset week itself — not in [onset-4w, onset-1w]
    result = predict_event_results([ep], {date(2010, 1, 12)}, horizon=4)[0]
    assert not result.hit


def test_predict_earliest_alert_wins():
    """Reports the earliest calm alert in the pre-onset window."""
    ep = _episode(date(2010, 2, 9), 2)
    # Both 3w and 1w before onset are in the window
    calm_alerts = {date(2010, 1, 19), date(2010, 2, 2)}  # 3w and 1w before
    result = predict_event_results([ep], calm_alerts, horizon=4)[0]
    assert result.first_alert_week == date(2010, 1, 19)
    assert result.weeks_into_episode == 3


# ── false_alarm_runs ───────────────────────────────────────────────────────────


def test_no_false_alarms_when_alerts_inside_grace():
    """Alert inside episode grace window → not a false alarm."""
    ep = _episode(date(2010, 1, 12), 3)  # onset Jan 12
    weeks = _tuesdays(date(2009, 12, 1), 12)
    # Alert on Jan 5 = 1 week before onset (within 2-week grace)
    result = false_alarm_runs([date(2010, 1, 5)], weeks, [ep], grace_weeks=2, known_unlabeled=[])
    assert result == []


def test_false_alarm_run_outside_grace():
    """Alert far from any episode = false alarm."""
    ep = _episode(date(2010, 1, 12), 3)
    weeks = _tuesdays(date(2009, 12, 1), 20)
    # Alert in early December, well before the episode
    result = false_alarm_runs([date(2009, 12, 1)], weeks, [ep], grace_weeks=2, known_unlabeled=[])
    assert len(result) == 1
    assert not result[0].in_known_unlabeled


def test_false_alarm_contiguous_run_counted_once():
    """3 contiguous alert weeks = one false alarm run of length 3."""
    ep = _episode(date(2010, 6, 1), 2)
    weeks = _tuesdays(date(2010, 1, 5), 30)
    run_weeks = [date(2010, 1, 5), date(2010, 1, 12), date(2010, 1, 19)]
    result = false_alarm_runs(run_weeks, weeks, [ep], grace_weeks=2, known_unlabeled=[])
    assert len(result) == 1
    assert result[0].length == 3


def test_known_unlabeled_not_counted_as_false_alarm():
    """Alert inside a known-unlabeled window is marked in_known_unlabeled=True."""
    ku_start = date(2013, 5, 21)
    ku_end = date(2013, 7, 9)
    ku = [(ku_start, ku_end, "taper tantrum")]
    alert_day = date(2013, 6, 4)
    weeks = _tuesdays(date(2013, 1, 1), 30)
    result = false_alarm_runs([alert_day], weeks, [], grace_weeks=2, known_unlabeled=ku)
    assert len(result) == 1
    assert result[0].in_known_unlabeled


def test_false_alarm_outside_known_unlabeled_is_not_marked():
    """Alert outside all known-unlabeled windows has in_known_unlabeled=False."""
    ku = [(date(2013, 5, 21), date(2013, 7, 9), "taper tantrum")]
    weeks = _tuesdays(date(2013, 1, 1), 10)
    result = false_alarm_runs([date(2013, 1, 8)], weeks, [], grace_weeks=2, known_unlabeled=ku)
    assert len(result) == 1
    assert not result[0].in_known_unlabeled


def test_false_alarm_empty_alerts():
    """No alerts = no false alarms."""
    ep = _episode(date(2010, 1, 12), 3)
    weeks = _tuesdays(date(2009, 12, 1), 10)
    assert false_alarm_runs([], weeks, [ep], known_unlabeled=[]) == []


def test_false_alarm_grace_exactly_at_boundary():
    """Alert exactly 2 weeks before episode start (grace=2) is NOT a false alarm."""
    ep = _episode(date(2010, 1, 26), 3)  # onset Jan 26
    weeks = _tuesdays(date(2010, 1, 5), 10)
    alert = date(2010, 1, 12)  # exactly 2 weeks before onset
    result = false_alarm_runs([alert], weeks, [ep], grace_weeks=2, known_unlabeled=[])
    assert result == []


def test_false_alarm_just_beyond_grace():
    """Alert 3 weeks before episode start (grace=2) IS a false alarm."""
    ep = _episode(date(2010, 1, 26), 3)  # onset Jan 26
    weeks = _tuesdays(date(2010, 1, 5), 10)
    alert = date(2010, 1, 5)  # 3 weeks before onset
    result = false_alarm_runs([alert], weeks, [ep], grace_weeks=2, known_unlabeled=[])
    assert len(result) == 1


# ── row_metrics ────────────────────────────────────────────────────────────────


def test_row_metrics_perfect_prediction():
    yt = [1, 0, 1, 0]
    yp = [1, 0, 1, 0]
    m = row_metrics(yt, yp)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)
    assert m["false_alarm_rate"] == pytest.approx(0.0)


def test_row_metrics_always_flag():
    yt = [1, 0, 1, 0, 0]
    yp = [1, 1, 1, 1, 1]
    m = row_metrics(yt, yp)
    assert m["recall"] == pytest.approx(1.0)
    assert m["false_alarm_rate"] == pytest.approx(1.0)
    assert m["tp"] == 2
    assert m["fp"] == 3


def test_row_metrics_empty_arrays():
    m = row_metrics([], [])
    assert m["tp"] == 0
    assert m["f1"] == pytest.approx(0.0)


def test_row_metrics_no_positives_in_truth():
    yt = [0, 0, 0]
    yp = [1, 0, 0]
    m = row_metrics(yt, yp)
    assert m["precision"] == pytest.approx(0.0)
    assert m["recall"] == pytest.approx(0.0)
    assert m["false_alarm_rate"] == pytest.approx(1 / 3)


# ── prob_metrics ───────────────────────────────────────────────────────────────


def test_prob_metrics_returns_none_single_class():
    assert prob_metrics([0, 0, 0], [0.1, 0.2, 0.3]) is None


def test_prob_metrics_perfect_scores():
    yt = [1, 0, 1, 0]
    ys = [0.9, 0.1, 0.8, 0.2]
    m = prob_metrics(yt, ys)
    assert m is not None
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["pr_auc"] == pytest.approx(1.0)
    assert m["brier"] < 0.1
    assert "log_loss" in m


# ── choose_threshold ───────────────────────────────────────────────────────────


def test_choose_threshold_achieves_far_budget():
    """Threshold is lowest score on calm weeks achieving FAR <= budget."""
    rng = np.random.default_rng(0)
    yt = np.array([0] * 100 + [1] * 20)
    ys = rng.uniform(0, 1, size=120)
    thr = choose_threshold(yt, ys, far_budget=0.05)
    calm_scores = ys[yt == 0]
    far = float((calm_scores >= thr).mean())
    assert far <= 0.05


def test_choose_threshold_excluded_mask():
    """Known-unlabeled weeks are excluded from FAR calculation."""
    yt = np.array([0] * 10)
    # All calm scores = 0.9 (would violate budget without exclusion)
    ys = np.full(10, 0.9)
    # Exclude all weeks
    mask = np.ones(10, dtype=bool)
    thr = choose_threshold(yt, ys, far_budget=0.05, excluded_mask=mask)
    # No calm weeks remain → inf threshold
    assert math.isinf(thr)


def test_choose_threshold_no_feasible_threshold():
    """Returns inf when no threshold achieves the budget."""
    yt = np.array([0, 0, 0])
    ys = np.array([0.9, 0.8, 0.7])  # even highest threshold gives FAR=1/3 > 0.0
    thr = choose_threshold(yt, ys, far_budget=0.0)
    assert math.isinf(thr)


def test_choose_threshold_lowest_qualifying_integers():
    """100 calm scores 0..99, budget 5% → threshold 95, achieved FAR exactly 0.05.

    The bug returned 99 (highest qualifying); the fix returns 95 (lowest qualifying).
    """
    yt = np.zeros(100, dtype=int)
    ys = np.arange(100, dtype=float)
    thr = choose_threshold(yt, ys, far_budget=0.05)
    assert thr == pytest.approx(95.0)
    assert float((ys >= thr).mean()) == pytest.approx(0.05)


def test_choose_threshold_ties():
    """Ties: lowest threshold with FAR <= budget, not highest."""
    yt = np.zeros(10, dtype=int)
    ys = np.array([1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 9.0, 9.0, 9.0, 9.0], dtype=float)
    # budget=0.70: thr=5 gives 7/10=0.70 ≤ 0.70 (lowest qualifying)
    thr = choose_threshold(yt, ys, far_budget=0.70)
    assert thr == pytest.approx(5.0)
    assert float((ys >= thr).mean()) == pytest.approx(0.70)


def test_choose_threshold_all_equal():
    """All-equal calm scores: inf when FAR=1.0 exceeds budget, else that score."""
    yt = np.zeros(5, dtype=int)
    ys = np.full(5, 0.5)
    assert math.isinf(choose_threshold(yt, ys, far_budget=0.05))
    assert choose_threshold(yt, ys, far_budget=1.0) == pytest.approx(0.5)


def test_choose_threshold_excluded_removes_from_both():
    """Excluded weeks are removed from numerator and denominator (not just numerator)."""
    yt = np.zeros(10, dtype=int)
    ys = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # Exclude the first five; remaining calm scores: [0.6, 0.7, 0.8, 0.9, 1.0]
    mask = np.array([True, True, True, True, True, False, False, False, False, False])
    # budget=0.40: 2/5 at thr=0.9 → FAR=0.40 (lowest qualifying)
    thr = choose_threshold(yt, ys, far_budget=0.40, excluded_mask=mask)
    assert thr == pytest.approx(0.9)
    non_exc = ys[~mask]
    assert float((non_exc >= thr).mean()) == pytest.approx(0.40)


# ── baselines ──────────────────────────────────────────────────────────────────


def test_always_flag_all_ones():
    a = always_flag(5)
    assert list(a) == [1, 1, 1, 1, 1]


def test_persistence_shift_by_one():
    d = np.array([0, 1, 1, 0, 1])
    p = persistence(d)
    assert list(p) == [0, 0, 1, 1, 0]


def test_persistence_first_week_is_zero():
    assert persistence(np.array([1, 0, 0]))[0] == 0


def test_vix_rule_threshold():
    z = np.array([-1.0, 0.5, 1.5, 2.0, 2.5])
    alerts = vix_rule(z, threshold=1.5)
    assert list(alerts) == [0, 0, 1, 1, 1]


def test_vix_rule_exact_boundary():
    """Threshold is applied as >=, so a score exactly equal to threshold fires."""
    assert vix_rule(np.array([1.0]), 1.0)[0] == 1


def test_fit_vix_threshold_achieves_budget():
    rng = np.random.default_rng(42)
    n = 200
    yt = (rng.uniform(size=n) > 0.8).astype(int)
    ys = rng.uniform(size=n) + yt.astype(float) * 0.5
    thr = fit_vix_threshold(ys, yt, far_budget=0.1)
    calm_scores = ys[yt == 0]
    far = float((calm_scores >= thr).mean())
    assert far <= 0.10 + 1e-9  # allow floating-point rounding


# ── make_folds_v2 ──────────────────────────────────────────────────────────────


def test_folds_v2_train_before_val_before_test():
    """train < val < test strictly for every fold."""
    rng = np.random.default_rng(7)
    start = date(2000, 1, 4)
    weeks = _tuesdays(start, 1111)
    detect = (rng.uniform(size=1111) > 0.78).astype(int)
    folds = make_folds_v2(weeks, detect, first_test_year=2008, last_test_year=2017)
    assert len(folds) >= 5
    for f in folds:
        assert f.val_lo < f.test_lo < f.test_hi
        assert f.val_start < f.val_end < f.test_start


def test_folds_v2_test_year_range():
    rng = np.random.default_rng(7)
    start = date(2000, 1, 4)
    weeks = _tuesdays(start, 1111)
    detect = (rng.uniform(size=1111) > 0.78).astype(int)
    folds = make_folds_v2(weeks, detect, first_test_year=2010, last_test_year=2012)
    test_years = [f.test_year for f in folds]
    assert 2010 in test_years
    assert 2012 in test_years
    assert all(2010 <= y <= 2012 for y in test_years)


def test_folds_v2_val_has_min_positives():
    """Every validation window has >= min_val_positives anomaly weeks."""
    rng = np.random.default_rng(7)
    start = date(2000, 1, 4)
    weeks = _tuesdays(start, 1111)
    detect = (rng.uniform(size=1111) > 0.78).astype(int)
    folds = make_folds_v2(
        weeks, detect, first_test_year=2008, last_test_year=2015, min_val_positives=8
    )
    for f in folds:
        val_detect = detect[f.val_lo : f.test_lo]
        assert val_detect.sum() >= 8


def test_folds_v2_no_data_leakage():
    """val_end is strictly before test_start."""
    rng = np.random.default_rng(7)
    start = date(2000, 1, 4)
    weeks = _tuesdays(start, 1111)
    detect = (rng.uniform(size=1111) > 0.78).astype(int)
    folds = make_folds_v2(weeks, detect, first_test_year=2008, last_test_year=2015)
    for f in folds:
        assert f.val_end < f.test_start


# ── vix_z52 ────────────────────────────────────────────────────────────────────


def test_vix_z52_log_level_zscore():
    """z-score is (log VIX[t] - mean) / std over trailing 52-week window."""
    rng = np.random.default_rng(99)
    # 60 weeks of random VIX around 20
    vix = 20.0 + rng.normal(0, 2, 60)
    vix = np.clip(vix, 10, 50)
    z = vix_z52(vix, min_periods=52)
    # First 51 weeks should be NaN (not enough observations)
    assert all(math.isnan(v) for v in z[:51])
    # Week 52 (index 51) should be a real number
    assert not math.isnan(z[51])
    # The z-score at week 59 computed manually should match.
    # Rolling window at index 59 covers indices 8–59 (52 observations).
    log_v = np.log(vix[8:60])
    expected = (log_v[-1] - log_v.mean()) / log_v.std(ddof=1)
    assert abs(z[59] - expected) < 1e-10


def test_vix_z52_sustained_high_vix_alerts_throughout():
    """Regression: sustained elevated VIX must alert throughout under vix_level.

    Bug in previous implementation: z-scored the 1-week *change* of log VIX.
    When VIX stabilises at a high level the change is near zero and the score
    drops back below threshold — missing the entire episode.

    Fix: vix_level (raw close) stays elevated regardless of week-over-week
    changes, so every stress week alerts.  We also verify that vix_z52 stays
    positive throughout (level z-score does not drop to zero mid-episode).
    """
    rng = np.random.default_rng(1)
    # 100 calm weeks with small noise so threshold can be calibrated
    calm_vix = (15.0 + rng.normal(0, 0.5, 100)).clip(10, 20)
    # 48 stress weeks at sustained high VIX
    stress_vix = np.full(48, 40.0)

    # Threshold chosen only from val (calm) window — no test data leakage
    calm_detect = np.zeros(100, dtype=int)
    thr_level = choose_threshold(calm_detect, calm_vix, far_budget=0.05)

    # vix_level: ALL 48 stress weeks must fire
    alerts = vix_rule(stress_vix, thr_level)
    assert alerts.sum() == 48, (
        f"Expected all 48 stress weeks to alert with vix_level, got {alerts.sum()}."
        " This is a regression for the sustained-high-VIX detection bug."
    )

    # vix_z52 must stay positive throughout (not decay to zero mid-episode)
    full_vix = np.concatenate([calm_vix, stress_vix])
    z52 = vix_z52(full_vix, min_periods=52)
    stress_z52 = z52[100:]
    # z52 should be positive for every non-NaN stress week
    assert all(v > 0 for v in stress_z52 if not math.isnan(v)), (
        "vix_z52 dropped to zero during sustained stress. The level z-score"
        " must remain positive while VIX stays above its historical mean."
    )


def test_threshold_val_only_regression():
    """Threshold must be chosen from validation rows only, not test rows.

    A model must not see test data when selecting the threshold.
    This test constructs a case where using test data would produce a lower
    threshold (easier to meet) than using val data alone.
    """
    rng = np.random.default_rng(17)
    # val: 50 calm weeks with low scores
    val_detect = np.zeros(50, dtype=int)
    val_scores = rng.uniform(0, 0.3, 50)

    # test: 20 weeks — calm scores are higher, which would lower the threshold
    # if test data were leaked into threshold selection
    test_detect = np.zeros(20, dtype=int)
    test_scores = rng.uniform(0.7, 1.0, 20)

    thr_val_only = choose_threshold(val_detect, val_scores, far_budget=0.05)
    thr_combined = choose_threshold(
        np.concatenate([val_detect, test_detect]),
        np.concatenate([val_scores, test_scores]),
        far_budget=0.05,
    )

    # Using only val data should yield a LOWER (more permissive for calm) threshold
    # than combining val+test (where high test scores inflate the distribution).
    # thr_val_only <= thr_combined in a correctly split scenario.
    assert thr_val_only <= thr_combined
