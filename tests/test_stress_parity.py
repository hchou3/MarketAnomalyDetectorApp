"""Offline unit tests for stress_parity alignment logic.

All tests use synthetic DataFrames — no database or network required.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from market_anomaly_detector.stress_parity import (
    align_cboe,
    align_fred,
    compare_column,
)

# ── helpers ───────────────────────────────────────────────────────────────────


def _prices(obs_dates: list[date], values: list[float], known_at: list[date] | None = None):
    if known_at is None:
        known_at = obs_dates
    return pd.DataFrame({"obs_date": obs_dates, "value": values, "known_at": known_at})


def _tuesdays(start: date, n: int) -> list[date]:
    return [start + timedelta(weeks=i) for i in range(n)]


_TUE = date(2020, 1, 7)  # a known Tuesday


# ── align_cboe: blank before series starts ────────────────────────────────────


def test_cboe_blank_before_start():
    """Tuesdays before the first obs_date must return NaN."""
    start = date(2020, 3, 3)  # Tuesday
    prices = _prices([start, start + timedelta(weeks=1)], [100.0, 101.0])
    tuesdays = [
        date(2020, 2, 4),  # before start
        date(2020, 2, 18),  # before start
        start,
        start + timedelta(weeks=1),
    ]
    result = align_cboe(prices, tuesdays)
    assert pd.isna(result[date(2020, 2, 4)])
    assert pd.isna(result[date(2020, 2, 18)])
    assert result[start] == pytest.approx(100.0)
    assert result[start + timedelta(weeks=1)] == pytest.approx(101.0)


def test_cboe_last_close_on_or_before_tuesday():
    """For each Tuesday, the last available obs_date on or before it is used."""
    # Data on Monday and Wednesday; Tuesdays fall between them
    obs = [date(2020, 1, 6), date(2020, 1, 8)]  # Mon, Wed
    vals = [50.0, 60.0]
    prices = _prices(obs, vals)
    tuesday = date(2020, 1, 7)  # Tuesday between the two obs_dates
    result = align_cboe(prices, [tuesday])
    # Only Monday (Jan 6) is ≤ Tuesday; Wednesday is in the future
    assert result[tuesday] == pytest.approx(50.0)


def test_cboe_carry_forward_across_holiday():
    """If no data on Tuesday, carry forward the prior obs_date (no tolerance limit)."""
    # Two weekly closes: Jan 7 and Jan 21 (skipping Jan 14)
    obs = [date(2020, 1, 7), date(2020, 1, 21)]
    prices = _prices(obs, [100.0, 102.0])
    # Tuesday Jan 14 is in the gap
    result = align_cboe(prices, [date(2020, 1, 7), date(2020, 1, 14), date(2020, 1, 21)])
    assert result[date(2020, 1, 7)] == pytest.approx(100.0)
    assert result[date(2020, 1, 14)] == pytest.approx(100.0)  # carried forward
    assert result[date(2020, 1, 21)] == pytest.approx(102.0)


def test_cboe_empty_prices_returns_all_nan():
    empty = pd.DataFrame(columns=["obs_date", "value", "known_at"])
    tuesdays = _tuesdays(_TUE, 5)
    result = align_cboe(empty, tuesdays)
    assert result.isna().all()


# ── align_fred: known_at publication lag ─────────────────────────────────────


def test_fred_known_at_lag_hides_unreleased_value():
    """A value not yet published (known_at > tuesday) must return NaN."""
    friday = date(2020, 1, 3)  # Friday obs_date
    known_at = friday + timedelta(days=11)  # Saturday Jan 14 → released Tue Jan 14
    prices = _prices([friday], [0.5], known_at=[known_at])

    tuesday_before = date(2020, 1, 7)  # Tuesday before known_at
    tuesday_on = known_at  # known_at is exactly Jan 14 (Tuesday)

    result_before = align_fred(prices, [tuesday_before])
    result_on = align_fred(prices, [tuesday_on])

    assert pd.isna(result_before[tuesday_before])
    assert result_on[tuesday_on] == pytest.approx(0.5)


def test_fred_known_at_lag_reveals_correct_value():
    """After known_at, the correct obs value is returned."""
    # Two Friday observations with 11-day lag
    obs = [date(2020, 1, 3), date(2020, 1, 10)]
    vals = [0.5, 0.6]
    known_at = [d + timedelta(days=11) for d in obs]
    prices = _prices(obs, vals, known_at=known_at)

    # First value: known Jan 14, second known Jan 21
    # Tuesday Jan 14: only first value is known
    # Tuesday Jan 21: both are known; latest obs (Jan 10) whose known_at ≤ Jan 21 → 0.6
    t1 = date(2020, 1, 14)
    t2 = date(2020, 1, 21)
    result = align_fred(prices, [t1, t2])
    assert result[t1] == pytest.approx(0.5)
    assert result[t2] == pytest.approx(0.6)


def test_fred_no_future_data_used():
    """Values whose known_at is after the Tuesday are never used."""
    future_friday = date(2020, 2, 7)
    known_at = future_friday + timedelta(days=11)
    prices = _prices([future_friday], [9.9], known_at=[known_at])

    earlier_tuesdays = _tuesdays(_TUE, 4)  # all before known_at
    result = align_fred(prices, earlier_tuesdays)
    assert result.isna().all()


def test_fred_empty_prices_returns_all_nan():
    empty = pd.DataFrame(columns=["obs_date", "value", "known_at"])
    result = align_fred(empty, _tuesdays(_TUE, 5))
    assert result.isna().all()


# ── compare_column ────────────────────────────────────────────────────────────


def _series(index: list[date], values: list[float | None]) -> pd.Series:
    return pd.Series(values, index=index, dtype=float)


def test_compare_identical_series():
    idx = _tuesdays(_TUE, 10)
    vals = [float(i) for i in range(10)]
    result = compare_column(_series(idx, vals), _series(idx, vals), "TEST", is_revised=False)
    assert result.matched == 10
    assert result.differing == 0
    assert result.missing == 0
    assert not result.fails


def test_compare_differing_non_revised_fails():
    idx = _tuesdays(_TUE, 5)
    db = _series(idx, [1.0, 2.0, 3.0, 4.0, 5.0])
    csv = _series(idx, [1.0, 2.0, 3.0, 4.0, 6.0])  # last differs
    result = compare_column(db, csv, "TEST", is_revised=False)
    assert result.differing == 1
    assert result.matched == 4
    assert result.fails


def test_compare_differing_revised_does_not_fail():
    idx = _tuesdays(_TUE, 5)
    db = _series(idx, [1.0, 2.1, 3.0, 4.0, 5.0])
    csv = _series(idx, [1.0, 2.0, 3.0, 4.0, 5.0])
    result = compare_column(db, csv, "NFCI", is_revised=True)
    assert result.differing == 1
    assert not result.fails


def test_compare_missing_non_revised_fails():
    idx = _tuesdays(_TUE, 5)
    db = _series(idx, [1.0, None, 3.0, 4.0, 5.0])  # DB missing week 2
    csv = _series(idx, [1.0, 2.0, 3.0, 4.0, 5.0])  # CSV has all
    result = compare_column(db, csv, "TEST", is_revised=False)
    assert result.missing == 1
    assert result.fails


def test_compare_missing_revised_does_not_fail():
    idx = _tuesdays(_TUE, 5)
    db = _series(idx, [None, None, None, None, None])
    csv = _series(idx, [1.0, 2.0, 3.0, 4.0, 5.0])
    result = compare_column(db, csv, "STLFSI", is_revised=True)
    assert result.missing == 5
    assert not result.fails


def test_compare_csv_nan_does_not_count_as_missing():
    """Weeks where the CSV itself is NaN (before series start) are not 'missing'."""
    idx = _tuesdays(_TUE, 6)
    # CSV has NaN for first 2 weeks (series not started yet)
    db = _series(idx, [None, None, 3.0, 4.0, 5.0, 6.0])
    csv = _series(idx, [None, None, 3.0, 4.0, 5.0, 6.0])
    result = compare_column(db, csv, "VIX3M", is_revised=False)
    assert result.missing == 0
    assert result.matched == 4
    assert not result.fails


def test_compare_max_abs_diff():
    idx = _tuesdays(_TUE, 4)
    db = _series(idx, [10.0, 20.0, 30.0, 40.0])
    csv = _series(idx, [10.0, 20.5, 30.0, 39.0])
    result = compare_column(db, csv, "TEST", is_revised=True)
    assert result.max_abs_diff == pytest.approx(1.0)


# ── Tuesday as-of logic (combined scenario) ───────────────────────────────────


def test_cboe_as_of_tuesday_uses_last_available():
    """Each Tuesday uses the last available Cboe close on or before it."""
    # Daily prices Mon-Thu surrounding two Tuesdays
    obs = [
        date(2020, 1, 6),  # Mon
        date(2020, 1, 8),  # Wed
        date(2020, 1, 9),  # Thu
        date(2020, 1, 13),  # Mon
    ]
    vals = [10.0, 12.0, 13.0, 15.0]
    prices = _prices(obs, vals)
    tuesdays = [date(2020, 1, 7), date(2020, 1, 14)]
    result = align_cboe(prices, tuesdays)
    # Jan 7: last obs ≤ Jan 7 is Jan 6 → 10.0
    assert result[date(2020, 1, 7)] == pytest.approx(10.0)
    # Jan 14: last obs ≤ Jan 14 is Jan 13 → 15.0
    assert result[date(2020, 1, 14)] == pytest.approx(15.0)


def test_fred_as_of_tuesday_respects_known_at():
    """FRED as-of Tuesday uses the most recent obs whose known_at ≤ tuesday."""
    # Three obs with staggered known_at
    obs = [date(2020, 1, 3), date(2020, 1, 10), date(2020, 1, 17)]
    vals = [0.1, 0.2, 0.3]
    known = [date(2020, 1, 14), date(2020, 1, 21), date(2020, 1, 28)]
    prices = _prices(obs, vals, known_at=known)

    tuesdays = [date(2020, 1, 14), date(2020, 1, 21), date(2020, 1, 28)]
    result = align_fred(prices, tuesdays)
    assert result[date(2020, 1, 14)] == pytest.approx(0.1)  # only first known by Jan 14
    assert result[date(2020, 1, 21)] == pytest.approx(0.2)  # second known by Jan 21
    assert result[date(2020, 1, 28)] == pytest.approx(0.3)  # third known by Jan 28
