"""Evaluation harness v2 for the Market Anomaly Detector.

Pure metric functions for row-level and event-level evaluation.
No database connections; all functions take DataFrames or arrays.

Row-level
---------
row_metrics(y_true, y_pred) -> dict
    P / R / F1 / false-alarm rate plus raw counts.

prob_metrics(y_true, y_score) -> dict | None
    Brier score, ROC AUC, PR AUC, log loss.  Returns None when fewer than
    two classes are present in y_true.

Event-level
-----------
detect_event_results(episodes, alert_set) -> list[EpisodeResult]
    Per-episode hit/miss; weeks_into_episode = 0 means onset week flagged.

predict_event_results(episodes, calm_alert_set, horizon) -> list[EpisodeResult]
    Advance-warning head evaluated on calm-at-t weeks only.
    weeks_into_episode is repurposed as weeks_before_onset (positive = advance).

false_alarm_runs(alert_weeks, all_weeks, episodes, grace_weeks, known_unlabeled)
    Contiguous alert runs that do not overlap any episode's grace window.
    Known-unlabeled event windows are flagged but not counted as false alarms.

Threshold
---------
choose_threshold(y_true, y_score, far_budget, excluded_mask) -> float
    Lowest threshold such that false-alarm rate ≤ far_budget on calm weeks
    (optionally excluding known-unlabeled windows from both numerator and
    denominator).

Baselines
---------
always_flag(n) -> np.ndarray
persistence(detect_labels) -> np.ndarray   [oracle reference — needs last week's label]
vix_z52(vix_close, min_periods) -> np.ndarray
    52-week rolling z-score of log VIX level.  Use with vix_rule + choose_threshold.
vix_rule(scores, threshold) -> np.ndarray
    Generic threshold rule: alert when score >= threshold.
fit_vix_threshold(scores, detect_labels, far_budget, excluded_mask) -> float

Folds
-----
make_folds_v2(weeks, detect, first_test_year, ...) -> list[FoldSpec]
    Expanding-window yearly folds; same logic as backtest.make_folds but
    works on date-indexed Series rather than a featured DataFrame.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from .episodes import Episode

# ── Known historical events that lack EWS labels ──────────────────────────────
# Alerts in these windows are reported but not counted as false alarms when
# selecting the detection threshold.
KNOWN_UNLABELED: list[tuple[date, date, str]] = [
    (date(2011, 2, 22), date(2011, 4, 5), "2011-03 Japan/Fukushima turbulence"),
    (date(2013, 5, 21), date(2013, 7, 9), "2013-06 Fed taper tantrum"),
    (date(2016, 6, 14), date(2016, 7, 19), "2016-06 Brexit vote"),
    (date(2018, 1, 30), date(2018, 3, 13), "2018-02 VIX spike (Volmageddon)"),
    (date(2019, 7, 30), date(2019, 9, 10), "2019-08 US–China trade-war escalation"),
]


# ── Dataclasses ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EpisodeResult:
    """Hit/miss result for one episode.

    For the detect head: ``weeks_into_episode`` = weeks from onset to first
    alert (0 = caught at onset, None = miss).
    For the predict head: ``weeks_into_episode`` is reinterpreted as
    ``weeks_before_onset`` = weeks of advance warning (1..h, None = miss).
    """

    episode: Episode
    hit: bool
    weeks_into_episode: int | None
    first_alert_week: date | None


@dataclass(frozen=True)
class FalseAlarmRun:
    """A contiguous run of alerts not overlapping any true episode's grace window."""

    start: date
    end: date
    length: int
    year: int
    in_known_unlabeled: bool


@dataclass(frozen=True)
class FoldSpec:
    """Row-index boundaries for one expanding-window fold."""

    test_year: int
    val_lo: int
    test_lo: int
    test_hi: int
    val_start: date
    val_end: date
    test_start: date
    test_end: date


# ── Event-level metrics ───────────────────────────────────────────────────────


def detect_event_results(
    episodes: list[Episode],
    alert_set: set[date],
) -> list[EpisodeResult]:
    """Per-episode hit/miss for the detect head.

    An episode is a hit if any week in ``alert_set`` falls within
    ``[ep.start, ep.end]``.  ``weeks_into_episode`` = weeks from onset to
    first alert inside the episode (0 = onset week flagged).
    """
    results: list[EpisodeResult] = []
    for ep in episodes:
        first = None
        d = ep.start
        while d <= ep.end:
            if d in alert_set:
                first = d
                break
            d += timedelta(weeks=1)
        if first is not None:
            weeks_in = (first - ep.start).days // 7
            results.append(EpisodeResult(ep, True, weeks_in, first))
        else:
            results.append(EpisodeResult(ep, False, None, None))
    return results


def predict_event_results(
    episodes: list[Episode],
    calm_alert_set: set[date],
    horizon: int,
) -> list[EpisodeResult]:
    """Per-episode advance-warning hit/miss for the predict head.

    A hit requires a calm-at-t alert in ``[ep.start − horizon weeks,
    ep.start − 1 week]`` (the pre-onset window).  ``weeks_into_episode``
    is repurposed as ``weeks_before_onset`` (positive = advance warning).

    Parameters
    ----------
    calm_alert_set:
        Weeks where the baseline fires AND ``detect[t] == 0`` (calm at t).
    """
    results: list[EpisodeResult] = []
    for ep in episodes:
        first = None
        for lag in range(horizon, 0, -1):  # check furthest ahead first
            candidate = ep.start - timedelta(weeks=lag)
            if candidate in calm_alert_set:
                first = candidate
                break
        if first is not None:
            weeks_before = (ep.start - first).days // 7
            results.append(EpisodeResult(ep, True, weeks_before, first))
        else:
            results.append(EpisodeResult(ep, False, None, None))
    return results


def false_alarm_runs(
    alert_weeks: Sequence[date],
    all_weeks: Sequence[date],
    true_episodes: list[Episode],
    grace_weeks: int = 2,
    known_unlabeled: list[tuple[date, date, str]] | None = None,
) -> list[FalseAlarmRun]:
    """Find contiguous alert runs that do not cover any true episode.

    A run overlaps an episode if any alert in the run falls within
    ``[ep.start − grace_weeks*7d, ep.end + grace_weeks*7d]``.  Such runs are
    excluded (they are near a real event).

    Runs that fall entirely within a known-unlabeled window are kept in the
    list (for reporting) but marked ``in_known_unlabeled = True``; callers
    exclude them from false-alarm budgets.
    """
    if known_unlabeled is None:
        known_unlabeled = KNOWN_UNLABELED

    alert_set = set(alert_weeks)
    weeks = list(all_weeks)

    # Expand episode grace windows.
    episode_grace: set[date] = set()
    for ep in true_episodes:
        d = ep.start - timedelta(weeks=grace_weeks)
        end = ep.end + timedelta(weeks=grace_weeks)
        while d <= end:
            episode_grace.add(d)
            d += timedelta(weeks=1)

    # Build contiguous runs from the sorted alert list.
    raw_runs: list[tuple[int, int]] = []
    n = len(weeks)
    i = 0
    while i < n:
        if weeks[i] in alert_set:
            j = i + 1
            while j < n and weeks[j] in alert_set:
                j += 1
            raw_runs.append((i, j - 1))
            i = j
        else:
            i += 1

    results: list[FalseAlarmRun] = []
    for si, ei in raw_runs:
        run_dates = [weeks[k] for k in range(si, ei + 1)]
        # Skip if any run date falls in a true episode's grace window.
        if any(d in episode_grace for d in run_dates):
            continue
        start = run_dates[0]
        end = run_dates[-1]
        length = (end - start).days // 7 + 1
        year = start.year

        # Check whether the run is entirely within a known-unlabeled window.
        in_ku = any(
            all(ku_start <= d <= ku_end for d in run_dates)
            for ku_start, ku_end, _ in known_unlabeled
        )
        results.append(
            FalseAlarmRun(start=start, end=end, length=length, year=year, in_known_unlabeled=in_ku)
        )

    return results


# ── Row-level metrics ─────────────────────────────────────────────────────────


def row_metrics(
    y_true: np.ndarray | list,
    y_pred: np.ndarray | list,
) -> dict:
    """Precision, recall, F1, false-alarm rate, plus TP/FP/FN/TN counts."""
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    tp = int(((yp == 1) & (yt == 1)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    tn = int(((yp == 0) & (yt == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    far = fp / (fp + tn) if fp + tn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "false_alarm_rate": far,
    }


def prob_metrics(
    y_true: np.ndarray | list,
    y_score: np.ndarray | list,
) -> dict | None:
    """Brier score, ROC AUC, PR AUC, log loss.

    Returns ``None`` if ``y_true`` contains only one class (undefined metrics).
    """
    yt = np.asarray(y_true, dtype=int)
    ys = np.asarray(y_score, dtype=float)
    if len(np.unique(yt)) < 2:
        return None
    return {
        "brier": float(brier_score_loss(yt, ys)),
        "roc_auc": float(roc_auc_score(yt, ys)),
        "pr_auc": float(average_precision_score(yt, ys)),
        "log_loss": float(log_loss(yt, ys)),
    }


# ── Threshold selection ───────────────────────────────────────────────────────


def choose_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    far_budget: float = 0.05,
    excluded_mask: np.ndarray | None = None,
) -> float:
    """Choose the lowest score threshold whose calm-week FAR ≤ *far_budget*.

    Parameters
    ----------
    y_true:
        Binary detect labels (0/1).
    y_score:
        Continuous scores (higher = more anomalous).
    excluded_mask:
        Boolean mask of weeks excluded from the FAR calculation (e.g.
        known-unlabeled event windows).  Excluded weeks are removed from
        both numerator and denominator.

    Returns
    -------
    float
        The chosen threshold, or ``np.inf`` if no threshold achieves the budget.
    """
    yt = np.asarray(y_true, dtype=int)
    ys = np.asarray(y_score, dtype=float)

    calm = yt == 0
    if excluded_mask is not None:
        calm = calm & ~np.asarray(excluded_mask, dtype=bool)

    calm_scores = ys[calm]
    if len(calm_scores) == 0:
        return float("inf")

    candidates = np.sort(np.unique(calm_scores))[::-1]
    for thr in candidates:
        far = float((calm_scores >= thr).mean())
        if far <= far_budget:
            return float(thr)
    return float("inf")


# ── Baselines ─────────────────────────────────────────────────────────────────


def always_flag(n: int) -> np.ndarray:
    """Return n ones (always-alert baseline)."""
    return np.ones(n, dtype=int)


def persistence(detect_labels: np.ndarray | list) -> np.ndarray:
    """Previous-week label (shift by 1; first week is 0)."""
    d = np.asarray(detect_labels, dtype=int)
    out = np.empty_like(d)
    out[0] = 0
    out[1:] = d[:-1]
    return out


def vix_z52(
    vix_close: np.ndarray | list,
    min_periods: int = 52,
) -> np.ndarray:
    """52-week rolling z-score of log VIX level.

    Score at week t = (log VIX[t] − mean(log VIX[t−51:t+1])) / std(...).
    Uses a 52-week trailing window (ddof=1).  Returns NaN until *min_periods*
    valid observations are available.

    A sustained high-VIX episode produces a persistently elevated score because
    the score is based on the *level* of log VIX, not the week-over-week change.
    """
    import pandas as pd

    vc = np.asarray(vix_close, dtype=float)
    log_vc = pd.Series(np.log(vc))
    mu = log_vc.rolling(52, min_periods=min_periods).mean()
    sigma = log_vc.rolling(52, min_periods=min_periods).std(ddof=1)
    result = (log_vc - mu) / sigma
    # When sigma==0 (constant series, sufficient data) set z=0 rather than NaN.
    # Do NOT touch positions where sigma is NaN (insufficient warmup data).
    constant_mask = (sigma == 0) & sigma.notna()
    result = result.where(~constant_mask, other=0.0)
    return result.to_numpy(dtype=float)


def vix_rule(scores: np.ndarray | list, threshold: float) -> np.ndarray:
    """Alert when *scores* meet or exceed *threshold* (generic threshold rule)."""
    z = np.asarray(scores, dtype=float)
    return (z >= threshold).astype(int)


def fit_vix_threshold(
    scores: np.ndarray,
    detect_labels: np.ndarray,
    far_budget: float = 0.05,
    excluded_mask: np.ndarray | None = None,
) -> float:
    """Choose the threshold for *scores* that achieves *far_budget* on the given window."""
    return choose_threshold(detect_labels, scores, far_budget, excluded_mask)


# ── Fold construction ─────────────────────────────────────────────────────────


def make_folds_v2(
    weeks: Sequence[date],
    detect: np.ndarray | list,
    first_test_year: int = 2008,
    last_test_year: int | None = None,
    val_years: int = 2,
    min_val_positives: int = 8,
    min_train_rows: int = 104,
) -> list[FoldSpec]:
    """Build expanding-window walk-forward folds.

    One fold per calendar test year from *first_test_year* to
    *last_test_year*.  The validation window is extended backwards until it
    contains at least *min_val_positives* anomaly weeks.  Train < val < test
    strictly.
    """
    ws = [w if isinstance(w, date) else w.date() for w in weeks]
    d = np.asarray(detect, dtype=int)
    years = np.array([w.year for w in ws])

    if last_test_year is None:
        last_test_year = int(years.max())

    folds: list[FoldSpec] = []
    for test_year in range(first_test_year, last_test_year + 1):
        test_pos = np.flatnonzero(years == test_year)
        if test_pos.size == 0:
            continue

        val_start_year = test_year - val_years
        first_year = int(years.min())
        while True:
            in_val = (years >= val_start_year) & (years < test_year)
            pos = int(d[in_val].sum())
            neg = int(in_val.sum() - pos)
            if pos >= min_val_positives and neg >= min_val_positives:
                break
            val_start_year -= 1
            if val_start_year <= first_year:
                raise ValueError(
                    f"Test year {test_year}: cannot find a validation window with "
                    f">= {min_val_positives} anomalies."
                )

        val_lo = int(np.flatnonzero(years >= val_start_year)[0])
        test_lo = int(test_pos[0])
        test_hi = int(test_pos[-1]) + 1

        if val_lo < min_train_rows:
            raise ValueError(f"Test year {test_year}: training window too small ({val_lo} rows).")

        folds.append(
            FoldSpec(
                test_year=test_year,
                val_lo=val_lo,
                test_lo=test_lo,
                test_hi=test_hi,
                val_start=ws[val_lo],
                val_end=ws[test_lo - 1],
                test_start=ws[test_lo],
                test_end=ws[test_hi - 1],
            )
        )
    return folds
