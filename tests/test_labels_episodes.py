"""Tests for labels.py and episodes.py on synthetic label data.

All tests are offline (no database required).
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pandas as pd
import pytest

from market_anomaly_detector.episodes import extract_episodes
from market_anomaly_detector.labels import build_targets

# ── Helpers ────────────────────────────────────────────────────────────────────


def _tuesdays(start: date, n: int) -> list[date]:
    """Return *n* consecutive Tuesdays starting on or after *start*."""
    d = start
    while d.weekday() != 1:
        d += timedelta(days=1)
    return [d + timedelta(weeks=i) for i in range(n)]


def _labels_df(values: list[int], start: date | None = None) -> pd.DataFrame:
    """Build a labels DataFrame from a list of 0/1 values."""
    if start is None:
        start = date(2020, 1, 7)  # a Tuesday
    weeks = _tuesdays(start, len(values))
    return pd.DataFrame({"week": weeks, "value": values})


# ── build_targets ──────────────────────────────────────────────────────────────


def test_build_targets_detect_equals_value():
    """detect is always the label at week t."""
    df = _labels_df([1, 0, 1, 0, 0, 1, 0, 0])
    targets = build_targets(df, horizon=4)
    assert list(targets["detect"]) == [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]


def test_build_targets_predict_looks_ahead():
    """predict = 1 if any of the next h weeks is anomalous."""
    # values: 0 0 0 0 1 0 0 0 0 0 (h=4)
    # week 0: look at weeks 1-4 → all 0 → predict=0
    # week 0 (0-indexed): weeks[1..4] = [0,0,0,1] → predict=1
    df = _labels_df([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    targets = build_targets(df, horizon=4)
    # week 0: looks at weeks 1,2,3,4 → week4=1 → predict=1
    assert targets["predict"].iloc[0] == pytest.approx(1.0)
    # week 1: looks at weeks 2,3,4,5 → week4=1 → predict=1
    assert targets["predict"].iloc[1] == pytest.approx(1.0)
    # week 5: looks at weeks 6,7,8,9 → all 0 → predict=0
    assert targets["predict"].iloc[5] == pytest.approx(0.0)


def test_build_targets_last_h_rows_are_nan():
    """The last h rows have predict=NaN (no ground truth exists yet)."""
    df = _labels_df([1, 0, 0, 1, 0, 0, 0, 1])  # 8 rows, h=4
    targets = build_targets(df, horizon=4)
    n = len(targets)
    h = 4
    # First n-h rows must be non-NaN
    for i in range(n - h):
        assert not math.isnan(targets["predict"].iloc[i]), f"row {i} should be valid"
    # Last h rows must be NaN
    for i in range(n - h, n):
        assert math.isnan(targets["predict"].iloc[i]), f"row {i} should be NaN"


def test_build_targets_predict_horizon_near_end():
    """predict stays valid until exactly the (n-h)th row, then goes NaN."""
    # 6 labels, h=4 → rows 0,1 valid; rows 2,3,4,5 are NaN
    df = _labels_df([0, 1, 0, 0, 1, 0])
    targets = build_targets(df, horizon=4)
    assert not math.isnan(targets["predict"].iloc[0])
    assert not math.isnan(targets["predict"].iloc[1])
    for i in range(2, 6):
        assert math.isnan(targets["predict"].iloc[i])


def test_build_targets_all_calm_predict_zero():
    """All-calm labels give predict=0 for valid rows."""
    df = _labels_df([0] * 10)
    targets = build_targets(df, horizon=4)
    for i in range(6):
        assert targets["predict"].iloc[i] == pytest.approx(0.0)
    for i in range(6, 10):
        assert math.isnan(targets["predict"].iloc[i])


def test_build_targets_empty_returns_empty():
    """Empty input produces an empty DataFrame with the right columns."""
    targets = build_targets(pd.DataFrame(columns=["week", "value"]))
    assert list(targets.columns) == ["week", "detect", "predict"]
    assert len(targets) == 0


def test_build_targets_configurable_horizon():
    """horizon parameter is respected: last horizon rows are NaN."""
    df = _labels_df([0, 1, 0, 0, 0, 0])
    for h in (1, 2, 3):
        targets = build_targets(df, horizon=h)
        n = len(targets)
        for i in range(n - h):
            assert not math.isnan(targets["predict"].iloc[i])
        for i in range(n - h, n):
            assert math.isnan(targets["predict"].iloc[i])


# ── extract_episodes ───────────────────────────────────────────────────────────


def test_extract_episodes_empty_labels():
    """Empty input returns an empty list."""
    assert extract_episodes(pd.DataFrame(columns=["week", "value"])) == []


def test_extract_episodes_all_calm():
    """All-calm labels return an empty list."""
    df = _labels_df([0, 0, 0, 0, 0])
    assert extract_episodes(df) == []


def test_extract_episodes_single_week():
    """A single anomaly week is a length-1 episode."""
    df = _labels_df([0, 0, 1, 0, 0])
    episodes = extract_episodes(df, merge_gap=2)
    assert len(episodes) == 1
    ep = episodes[0]
    assert ep.start == ep.end == ep.peak_week
    assert ep.length == 1


def test_extract_episodes_no_merge_three_separate():
    """Three isolated single-week anomalies become three episodes."""
    df = _labels_df([1, 0, 0, 1, 0, 0, 1])
    episodes = extract_episodes(df, merge_gap=0)
    assert len(episodes) == 3
    for ep in episodes:
        assert ep.length == 1


def test_extract_episodes_merge_gap_exact_boundary_merges():
    """Two runs separated by exactly merge_gap calm weeks are merged."""
    # [1,1,0,0,1,1] — gap of 2 calm weeks between two 2-week runs
    df = _labels_df([1, 1, 0, 0, 1, 1])
    episodes = extract_episodes(df, merge_gap=2)
    assert len(episodes) == 1
    ep = episodes[0]
    assert ep.length == 6  # spans all 6 weeks


def test_extract_episodes_merge_gap_one_over_does_not_merge():
    """Two runs separated by merge_gap+1 calm weeks are NOT merged."""
    # [1,1,0,0,0,1,1] — gap of 3 calm weeks, merge_gap=2 → two episodes
    df = _labels_df([1, 1, 0, 0, 0, 1, 1])
    episodes = extract_episodes(df, merge_gap=2)
    assert len(episodes) == 2
    assert episodes[0].length == 2
    assert episodes[1].length == 2


def test_extract_episodes_episode_at_start_of_series():
    """An episode starting at the first week is detected correctly."""
    df = _labels_df([1, 1, 1, 0, 0])
    episodes = extract_episodes(df, merge_gap=2)
    assert len(episodes) == 1
    assert episodes[0].start == df["week"].iloc[0]
    assert episodes[0].length == 3


def test_extract_episodes_episode_at_end_of_series():
    """An episode ending at the last week is detected correctly."""
    df = _labels_df([0, 0, 1, 1, 1])
    episodes = extract_episodes(df, merge_gap=2)
    assert len(episodes) == 1
    assert episodes[0].end == df["week"].iloc[-1]
    assert episodes[0].length == 3


def test_extract_episodes_episode_spanning_full_series():
    """All-anomaly labels produce one episode spanning the full series."""
    df = _labels_df([1, 1, 1, 1, 1])
    episodes = extract_episodes(df, merge_gap=2)
    assert len(episodes) == 1
    assert episodes[0].start == df["week"].iloc[0]
    assert episodes[0].end == df["week"].iloc[-1]
    assert episodes[0].length == 5


def test_extract_episodes_merge_gap_zero_no_bridging():
    """merge_gap=0 never bridges calm weeks; adjacent anomaly runs stay separate."""
    # [1,1,0,1,1] — one calm week gap; with merge_gap=0, that should NOT merge
    df = _labels_df([1, 1, 0, 1, 1])
    episodes = extract_episodes(df, merge_gap=0)
    assert len(episodes) == 2


def test_extract_episodes_merge_gap_zero_adjacent_merges():
    """merge_gap=0: directly adjacent runs (no gap) still form one episode."""
    df = _labels_df([1, 1, 1, 0, 0, 1])
    episodes = extract_episodes(df, merge_gap=0)
    assert len(episodes) == 2
    assert episodes[0].length == 3
    assert episodes[1].length == 1


def test_extract_episodes_peak_week_is_start():
    """peak_week equals the episode start (onset) for binary-label episodes."""
    df = _labels_df([0, 1, 1, 0, 1])
    episodes = extract_episodes(df, merge_gap=2)
    for ep in episodes:
        assert ep.peak_week == ep.start


def test_extract_episodes_length_includes_bridged_gap():
    """Episode length counts all weeks in the span, including bridged calm weeks."""
    # Three anomaly weeks, 2 calm, three anomaly weeks → one episode of length 8
    df = _labels_df([1, 1, 1, 0, 0, 1, 1, 1])
    episodes = extract_episodes(df, merge_gap=2)
    assert len(episodes) == 1
    assert episodes[0].length == 8


def test_extract_episodes_multiple_episodes_order():
    """Episodes are returned in chronological order."""
    # Two isolated 1-week episodes
    df = _labels_df([1, 0, 0, 0, 1])
    episodes = extract_episodes(df, merge_gap=0)
    assert len(episodes) == 2
    assert episodes[0].start < episodes[1].start
