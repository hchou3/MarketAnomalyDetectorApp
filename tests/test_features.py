"""Tests for the feature spec and builder.

All tests are offline (no database required).  Synthetic price DataFrames with
known values are used to verify each transform and the point-in-time guarantee.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from market_anomaly_detector.features.build import (
    _drawdown,
    _log_return,
    _rvol,
    _spread,
    _weekly_series,
    _yield_change,
    _z_score,
    build,
)
from market_anomaly_detector.features.spec import SPEC_V1, SPEC_V2, FeatureSpec, spec_hash

# ── Helpers ───────────────────────────────────────────────────────────────────


def _tuesdays(start: date, n: int) -> list[date]:
    """Return *n* consecutive Tuesdays starting on or after *start*."""
    d = start
    while d.weekday() != 1:  # 1 = Tuesday
        d += timedelta(days=1)
    return [d + timedelta(weeks=i) for i in range(n)]


def _price_df(
    obs_dates: list[date],
    values: list[float],
    lag_days: int = 0,
) -> pd.DataFrame:
    """Build a minimal prices DataFrame with known_at = obs_date + lag_days."""
    known_at = [d + timedelta(days=lag_days) for d in obs_dates]
    return pd.DataFrame({"obs_date": obs_dates, "value": values, "known_at": known_at})


def _daily_price_df(start: date, n_days: int, value: float = 100.0) -> pd.DataFrame:
    """Constant-price daily series over n_days business days starting on start."""
    idx = pd.bdate_range(start, periods=n_days)
    return pd.DataFrame(
        {
            "obs_date": [d.date() for d in idx],
            "value": [value] * n_days,
            "known_at": [d.date() for d in idx],
        }
    )


# ── Weekly-series alignment ───────────────────────────────────────────────────


def test_weekly_series_uses_latest_known_at_value():
    """The last value with known_at <= Tuesday is returned for each Tuesday."""
    tuesdays = _tuesdays(date(2020, 1, 1), 3)
    # Tuesday 2020-01-07, 2020-01-14, 2020-01-21
    t0, t1, t2 = tuesdays

    df = _price_df(
        obs_dates=[date(2020, 1, 6), date(2020, 1, 13), date(2020, 1, 20)],
        values=[10.0, 11.0, 12.0],
        lag_days=0,
    )
    tues_idx = pd.DatetimeIndex([pd.Timestamp(t) for t in tuesdays])
    s = _weekly_series(df, tues_idx)

    assert s[pd.Timestamp(t0)] == pytest.approx(10.0)
    assert s[pd.Timestamp(t1)] == pytest.approx(11.0)
    assert s[pd.Timestamp(t2)] == pytest.approx(12.0)


def test_weekly_series_publication_lag_hides_unreleased_value():
    """A value with known_at > Tuesday is not visible on that Tuesday."""
    tuesdays = _tuesdays(date(2020, 1, 1), 2)
    t0, t1 = tuesdays  # 2020-01-07, 2020-01-14

    # obs 2020-01-07, but not published until 2020-01-18 (after t1)
    df = _price_df(
        obs_dates=[date(2020, 1, 6), date(2020, 1, 7)],
        values=[9.0, 10.0],
        lag_days=0,
    )
    df.loc[df["obs_date"] == date(2020, 1, 7), "known_at"] = date(2020, 1, 18)

    tues_idx = pd.DatetimeIndex([pd.Timestamp(t) for t in tuesdays])
    s = _weekly_series(df, tues_idx)

    # On t0 (2020-01-07): known_at for 2020-01-07 row is 2020-01-18 > t0 → not visible.
    # Only 2020-01-06 (known_at = 2020-01-06) is available; gap = 1 day ≤ 7d daily tolerance.
    assert s[pd.Timestamp(t0)] == pytest.approx(9.0)
    # On t1 (2020-01-14): the Jan 7 obs is still hidden (known_at = 2020-01-18 > t1).
    # The Jan 6 obs (known_at = 2020-01-06) is 8 days stale by Jan 14, exceeding the
    # 7-day daily tolerance, so the result is NaN rather than a stale carry-forward.
    assert math.isnan(s[pd.Timestamp(t1)])


def test_weekly_series_publication_lag_reveals_value_when_known():
    """A lagged value becomes visible once its known_at passes."""
    tuesdays = _tuesdays(date(2020, 1, 1), 3)
    t0, t1, t2 = tuesdays  # 2020-01-07, 2020-01-14, 2020-01-21

    # obs 2020-01-10, known_at 2020-01-21 (after t1 but <= t2)
    df = _price_df(
        obs_dates=[date(2020, 1, 6), date(2020, 1, 10)],
        values=[9.0, 15.0],
        lag_days=0,
    )
    df.loc[df["obs_date"] == date(2020, 1, 10), "known_at"] = date(2020, 1, 21)

    tues_idx = pd.DatetimeIndex([pd.Timestamp(t) for t in tuesdays])
    s = _weekly_series(df, tues_idx)

    assert s[pd.Timestamp(t0)] == pytest.approx(9.0)
    assert s[pd.Timestamp(t1)] == pytest.approx(9.0)  # known_at 2020-01-21 > t1
    assert s[pd.Timestamp(t2)] == pytest.approx(15.0)  # known_at 2020-01-21 <= t2


def test_weekly_series_carry_forward_across_gap():
    """With no new data, the last known value carries forward."""
    tuesdays = _tuesdays(date(2020, 1, 1), 4)
    t0, t1, t2, t3 = tuesdays

    df = _price_df(
        obs_dates=[date(2020, 1, 6), date(2020, 1, 27)],  # gap for t1, t2
        values=[10.0, 12.0],
        lag_days=0,
    )
    tues_idx = pd.DatetimeIndex([pd.Timestamp(t) for t in tuesdays])
    s = _weekly_series(df, tues_idx)

    assert s[pd.Timestamp(t0)] == pytest.approx(10.0)
    assert s[pd.Timestamp(t1)] == pytest.approx(10.0)  # carries forward
    assert s[pd.Timestamp(t2)] == pytest.approx(10.0)
    assert s[pd.Timestamp(t3)] == pytest.approx(12.0)


def test_weekly_series_empty_returns_nan():
    tuesdays = _tuesdays(date(2020, 1, 1), 3)
    tues_idx = pd.DatetimeIndex([pd.Timestamp(t) for t in tuesdays])
    s = _weekly_series(pd.DataFrame(columns=["obs_date", "value", "known_at"]), tues_idx)
    assert s.isna().all()


# ── Transform unit tests ──────────────────────────────────────────────────────


def test_log_return_1w():
    tuesdays = _tuesdays(date(2020, 1, 1), 3)
    t0, t1, t2 = tuesdays
    tues_idx = pd.DatetimeIndex([pd.Timestamp(t) for t in tuesdays])
    s = pd.Series([100.0, 110.0, 121.0], index=tues_idx, dtype=float)

    r = _log_return(s, 1)
    assert math.isnan(r.iloc[0])
    assert r.iloc[1] == pytest.approx(math.log(110.0 / 100.0))
    assert r.iloc[2] == pytest.approx(math.log(121.0 / 110.0))


def test_log_return_multiweek():
    tuesdays = _tuesdays(date(2020, 1, 1), 5)
    tues_idx = pd.DatetimeIndex([pd.Timestamp(t) for t in tuesdays])
    prices = [100.0, 102.0, 105.0, 103.0, 108.0]
    s = pd.Series(prices, index=tues_idx, dtype=float)

    r4 = _log_return(s, 4)
    assert math.isnan(r4.iloc[0])
    assert math.isnan(r4.iloc[3])
    assert r4.iloc[4] == pytest.approx(math.log(108.0 / 100.0))


def test_log_return_negative_price_is_nan():
    """WTI went negative in 2020-04; log return should propagate NaN."""
    tuesdays = _tuesdays(date(2020, 4, 14), 2)
    tues_idx = pd.DatetimeIndex([pd.Timestamp(t) for t in tuesdays])
    s = pd.Series([20.0, -37.63], index=tues_idx, dtype=float)

    r = _log_return(s, 1)
    assert math.isnan(r.iloc[1])


def test_yield_change_bp():
    tuesdays = _tuesdays(date(2020, 1, 1), 3)
    tues_idx = pd.DatetimeIndex([pd.Timestamp(t) for t in tuesdays])
    # yields in percent: 2.50%, 2.55%, 2.40%
    s = pd.Series([2.50, 2.55, 2.40], index=tues_idx, dtype=float)

    yc = _yield_change(s, 1)
    assert math.isnan(yc.iloc[0])
    assert yc.iloc[1] == pytest.approx(5.0)  # +5 bp
    assert yc.iloc[2] == pytest.approx(-15.0)  # -15 bp


def test_yield_change_multiweek():
    tuesdays = _tuesdays(date(2020, 1, 1), 5)
    tues_idx = pd.DatetimeIndex([pd.Timestamp(t) for t in tuesdays])
    s = pd.Series([2.00, 2.10, 2.20, 2.30, 2.40], index=tues_idx, dtype=float)

    yc4 = _yield_change(s, 4)
    assert math.isnan(yc4.iloc[3])
    assert yc4.iloc[4] == pytest.approx(40.0)  # 2.40 - 2.00 = 0.40% = 40 bp


def test_drawdown_at_peak_is_zero():
    """Drawdown when at an all-time high within the window is 0."""
    tuesdays = _tuesdays(date(2020, 1, 1), 4)
    tues_idx = pd.DatetimeIndex([pd.Timestamp(t) for t in tuesdays])
    s = pd.Series([100.0, 110.0, 120.0, 130.0], index=tues_idx, dtype=float)

    d = _drawdown(s, 52)
    assert (d <= 1e-10).all()  # always at or near new high


def test_drawdown_correct_value():
    """After a 10% drop from peak, drawdown ≈ log(0.9)."""
    tuesdays = _tuesdays(date(2020, 1, 1), 3)
    tues_idx = pd.DatetimeIndex([pd.Timestamp(t) for t in tuesdays])
    s = pd.Series([100.0, 110.0, 99.0], index=tues_idx, dtype=float)

    d = _drawdown(s, 52)
    assert d.iloc[2] == pytest.approx(math.log(99.0 / 110.0), rel=1e-6)


def test_drawdown_window_limits_history():
    """With window=2, the 52-week peak from earlier doesn't count."""
    tuesdays = _tuesdays(date(2020, 1, 1), 4)
    tues_idx = pd.DatetimeIndex([pd.Timestamp(t) for t in tuesdays])
    # t0=200, t1=100, t2=90, t3=95 — t0 is not in the window-2 look-back for t3
    s = pd.Series([200.0, 100.0, 90.0, 95.0], index=tues_idx, dtype=float)

    d = _drawdown(s, 2)
    # At t3, window of 2 = {t2, t3} = {90, 95}, max = 95 → drawdown = 0
    assert d.iloc[3] == pytest.approx(0.0, abs=1e-9)


def test_spread_in_basis_points():
    tuesdays = _tuesdays(date(2020, 1, 1), 3)
    tues_idx = pd.DatetimeIndex([pd.Timestamp(t) for t in tuesdays])
    s1 = pd.Series([4.50, 4.60, 4.40], index=tues_idx, dtype=float)  # 10Y
    s2 = pd.Series([3.00, 3.10, 3.20], index=tues_idx, dtype=float)  # 3M

    sp = _spread(s1, s2)
    assert sp.iloc[0] == pytest.approx(150.0)  # 1.50% = 150 bp
    assert sp.iloc[1] == pytest.approx(150.0)
    assert sp.iloc[2] == pytest.approx(120.0)  # 1.20% = 120 bp


def test_spread_inverted_is_negative():
    tuesdays = _tuesdays(date(2020, 1, 1), 2)
    tues_idx = pd.DatetimeIndex([pd.Timestamp(t) for t in tuesdays])
    s1 = pd.Series([2.00, 1.80], index=tues_idx, dtype=float)  # 10Y
    s2 = pd.Series([2.50, 2.60], index=tues_idx, dtype=float)  # 3M

    sp = _spread(s1, s2)
    assert sp.iloc[0] == pytest.approx(-50.0)
    assert sp.iloc[1] == pytest.approx(-80.0)


def test_z_score_nan_before_min_periods():
    """Z-scores should be NaN when fewer than 104 weeks of return history exist."""
    from market_anomaly_detector.features.build import _Z_MIN_PERIODS

    n = _Z_MIN_PERIODS + 20
    tuesdays = _tuesdays(date(2005, 1, 4), n)
    tues_idx = pd.DatetimeIndex([pd.Timestamp(t) for t in tuesdays])
    rng = np.random.default_rng(42)
    prices = np.cumprod(1 + rng.normal(0, 0.01, n)) * 100
    s = pd.Series(prices, index=tues_idx, dtype=float)

    z = _z_score(s, "price")
    # The first (min_periods - 1) return values can't form a full window
    # returns start at index 1 (shift(1)); z_score uses rolling(156, min=104)
    # so rows 1..103 should be NaN (only 1..103 returns available)
    assert z.iloc[:_Z_MIN_PERIODS].isna().all()
    # Once we have >= 104 returns (row 104), we should get a number
    assert not math.isnan(z.iloc[_Z_MIN_PERIODS])


def test_z_score_mean_near_zero_for_stationary_returns():
    """Long flat series: z-score of a zero return should be near zero."""
    n = 300
    tuesdays = _tuesdays(date(2000, 1, 4), n)
    tues_idx = pd.DatetimeIndex([pd.Timestamp(t) for t in tuesdays])
    # Constant price → all returns are 0 → std=0 → z-score NaN
    s = pd.Series(100.0, index=tues_idx, dtype=float)
    z = _z_score(s, "price")
    # All returns are 0, std=0, z should be NaN (0/0)
    assert z.dropna().empty or z.dropna().abs().max() < 1e-6


def test_z_score_percent_units():
    """For percent units, z-score is based on yield changes (bp)."""
    n = 250
    tuesdays = _tuesdays(date(2000, 1, 4), n)
    tues_idx = pd.DatetimeIndex([pd.Timestamp(t) for t in tuesdays])
    rng = np.random.default_rng(7)
    yields = 2.5 + np.cumsum(rng.normal(0, 0.02, n))
    s = pd.Series(yields, index=tues_idx, dtype=float)

    z = _z_score(s, "percent")
    # Should produce finite values for later rows
    late = z.iloc[-50:]
    assert late.notna().sum() >= 40


def test_rvol_constant_price_is_zero():
    """A series with constant prices has log returns of exactly 0 → rvol = 0."""
    d_start = date(2020, 1, 2)
    daily_df = _daily_price_df(d_start, 60, value=100.0)

    tuesdays = _tuesdays(d_start + timedelta(days=21), 3)
    tues_idx = pd.DatetimeIndex([pd.Timestamp(t) for t in tuesdays])
    rv = _rvol(daily_df, tues_idx, 20)
    # All log returns are 0 → std = 0
    valid = rv.dropna()
    assert len(valid) >= 1
    assert valid.abs().max() == pytest.approx(0.0, abs=1e-9)


def test_rvol_known_volatility():
    """Daily returns drawn from N(0, σ) → rvol ≈ σ * sqrt(252)."""
    n_days = 300
    daily_sigma = 0.01
    rng = np.random.default_rng(0)
    rets = rng.normal(0, daily_sigma, n_days)
    prices = np.cumprod(1 + rets) * 100

    d_start = date(2000, 1, 3)
    idx = pd.bdate_range(d_start, periods=n_days)
    daily_df = pd.DataFrame(
        {
            "obs_date": [d.date() for d in idx],
            "value": prices,
            "known_at": [d.date() for d in idx],
        }
    )
    tuesdays = _tuesdays(d_start + timedelta(days=120), 5)
    tues_idx = pd.DatetimeIndex([pd.Timestamp(t) for t in tuesdays])
    rv = _rvol(daily_df, tues_idx, 20)

    expected = daily_sigma * math.sqrt(252)
    valid = rv.dropna()
    assert len(valid) >= 3
    # With 20-day window the estimate has variance; allow 50% relative error
    assert valid.mean() == pytest.approx(expected, rel=0.5)


def test_rvol_stale_data_returns_nan():
    """If the daily series ends > _RVOL_STALE_DAYS before a Tuesday, return NaN."""
    from market_anomaly_detector.features.build import _RVOL_STALE_DAYS

    d_start = date(2020, 1, 2)
    daily_df = _daily_price_df(d_start, 40)

    # Tuesday well after the daily series ends
    last_data = date(2020, 2, 28)
    far_tuesday = last_data + timedelta(days=_RVOL_STALE_DAYS + 7)
    while far_tuesday.weekday() != 1:
        far_tuesday += timedelta(days=1)
    tues_idx = pd.DatetimeIndex([pd.Timestamp(far_tuesday)])
    rv = _rvol(daily_df, tues_idx, 20)
    assert math.isnan(rv.iloc[0])


# ── Full build() tests ────────────────────────────────────────────────────────


def _make_prices_for_spec(
    spec: list[FeatureSpec],
    tuesdays: list[date],
    extra_weeks: int = 200,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Create synthetic price and daily DataFrames for every instrument in spec."""
    rng = np.random.default_rng(seed)
    instrument_ids: set[str] = set()
    for feat in spec:
        instrument_ids.add(feat.instrument_id)
        if feat.instrument_id_2:
            instrument_ids.add(feat.instrument_id_2)

    prices: dict[str, pd.DataFrame] = {}
    for iid in instrument_ids:
        n_weeks = len(tuesdays) + extra_weeks
        first_tuesday = tuesdays[0] - timedelta(weeks=extra_weeks)
        while first_tuesday.weekday() != 1:
            first_tuesday += timedelta(days=1)
        all_tuesdays = [first_tuesday + timedelta(weeks=i) for i in range(n_weeks)]

        units = next((f.units for f in spec if f.instrument_id == iid), "price")
        if units == "percent":
            vals = 2.5 + np.cumsum(rng.normal(0, 0.02, n_weeks))
        else:
            vals = np.cumprod(1 + rng.normal(0, 0.01, n_weeks)) * 100.0

        obs_dates = []
        values = []
        known_ats = []
        for t, v in zip(all_tuesdays, vals):
            # One daily obs per Tuesday; known_at = obs_date (no lag for synthetic)
            obs_dates.append(t)
            values.append(float(v))
            known_ats.append(t)

        prices[iid] = pd.DataFrame({"obs_date": obs_dates, "value": values, "known_at": known_ats})
    return prices


def test_build_returns_correct_shape():
    spec = [
        FeatureSpec("p_lr_1w", "P 1w", "p", "price", "log_return", 1),
        FeatureSpec("p_lr_4w", "P 4w", "p", "price", "log_return", 4),
    ]
    tuesdays = _tuesdays(date(2020, 1, 7), 10)
    prices = _make_prices_for_spec(spec, tuesdays)
    df = build(prices, spec, tuesdays)

    assert df.shape == (10, 2)
    assert list(df.columns) == ["p_lr_1w", "p_lr_4w"]
    assert isinstance(df.index, pd.DatetimeIndex)


def test_build_missing_instrument_returns_nan_column():
    spec = [FeatureSpec("x_lr_1w", "X 1w", "x", "price", "log_return", 1)]
    tuesdays = _tuesdays(date(2020, 1, 7), 5)
    df = build({}, spec, tuesdays)  # no prices supplied

    assert df["x_lr_1w"].isna().all()


def test_build_is_deterministic():
    spec = SPEC_V1[:10]
    tuesdays = _tuesdays(date(2005, 1, 4), 50)
    prices = _make_prices_for_spec(spec, tuesdays)

    df1 = build(prices, spec, tuesdays)
    df2 = build(prices, spec, tuesdays)
    pd.testing.assert_frame_equal(df1, df2)


def test_build_as_of_masks_future_prices():
    """build(..., as_of=T) must not use any price with known_at > T."""
    spec = [FeatureSpec("p_lr_1w", "P 1w", "p", "price", "log_return", 1)]
    tuesdays = _tuesdays(date(2020, 1, 7), 5)
    t0, t1, t2, t3, t4 = tuesdays

    # All prices have known_at == obs_date
    prices = {"p": _price_df(list(tuesdays), [100.0, 110.0, 105.0, 112.0, 108.0])}

    # as_of = t2: mask to known_at <= t2, so only t0,t1,t2 visible
    df_asof = build(prices, spec, tuesdays, as_of=t2)

    # Features requiring t3 or t4 data must be NaN
    assert math.isnan(df_asof.loc[pd.Timestamp(t3), "p_lr_1w"])
    assert math.isnan(df_asof.loc[pd.Timestamp(t4), "p_lr_1w"])
    # Feature at t1 requires t0 and t1 prices, both visible
    assert not math.isnan(df_asof.loc[pd.Timestamp(t1), "p_lr_1w"])


def test_build_spread():
    spec = [
        FeatureSpec(
            "spread_10_3", "10y-3m spread", "y10", "percent", "spread", 0, instrument_id_2="y3m"
        )
    ]
    tuesdays = _tuesdays(date(2020, 1, 7), 3)
    prices = {
        "y10": _price_df(list(tuesdays), [4.5, 4.6, 4.4]),
        "y3m": _price_df(list(tuesdays), [3.0, 3.1, 3.2]),
    }
    df = build(prices, spec, tuesdays)

    assert df["spread_10_3"].iloc[0] == pytest.approx(150.0)
    assert df["spread_10_3"].iloc[2] == pytest.approx(120.0)


def test_build_drawdown():
    spec = [FeatureSpec("dd_52", "52w drawdown", "p", "price", "drawdown", 52)]
    tuesdays = _tuesdays(date(2020, 1, 7), 4)
    prices = {"p": _price_df(list(tuesdays), [100.0, 110.0, 120.0, 108.0])}
    df = build(prices, spec, tuesdays)

    # At t3=108 after peak of 120: drawdown = log(108/120)
    assert df["dd_52"].iloc[3] == pytest.approx(math.log(108.0 / 120.0), rel=1e-6)


def test_build_wti_negative_price_propagates_nan():
    """WTI going negative causes NaN in log-return features for that week."""
    spec = [FeatureSpec("wti_lr_1w", "WTI 1w", "wti_front", "price", "log_return", 1)]
    tuesdays = _tuesdays(date(2020, 4, 14), 3)
    prices = {"wti_front": _price_df(list(tuesdays), [20.0, -37.63, 15.0])}
    df = build(prices, spec, tuesdays)

    assert math.isnan(df["wti_lr_1w"].iloc[1])
    # Recovery: log return from negative is also NaN since base is negative
    assert math.isnan(df["wti_lr_1w"].iloc[2])


def test_build_unknown_transform_raises():
    spec = [FeatureSpec("bad", "bad", "p", "price", "nonexistent", 1)]
    tuesdays = _tuesdays(date(2020, 1, 7), 3)
    prices = {"p": _price_df(list(tuesdays), [100.0, 101.0, 102.0])}
    with pytest.raises(ValueError, match="Unknown transform"):
        build(prices, spec, tuesdays)


# ── No-look-ahead test ────────────────────────────────────────────────────────


def test_no_look_ahead():
    """Features for week t from the full data set equal those from data truncated at t.

    This is the key point-in-time correctness test.  We pick a subset of spec
    entries that cover all transform types and verify row-level equality.
    """
    spec = [
        FeatureSpec("p_lr_1w", "P 1w", "p", "price", "log_return", 1),
        FeatureSpec("p_lr_4w", "P 4w", "p", "price", "log_return", 4),
        FeatureSpec("p_drawdown", "P dd", "p", "price", "drawdown", 52),
        FeatureSpec("p_z", "P z", "p", "price", "z_score", 156),
        FeatureSpec("y_yc_1w", "Y 1w", "y", "percent", "yield_change", 1),
        FeatureSpec("y_z", "Y z", "y", "percent", "z_score", 156),
        FeatureSpec("sp", "spread", "y", "percent", "spread", 0, instrument_id_2="y2"),
    ]

    n = 500
    tuesdays = _tuesdays(date(2000, 1, 4), n)
    prices = _make_prices_for_spec(spec, tuesdays, extra_weeks=200, seed=99)

    # Add a y2 instrument (for spread)
    rng = np.random.default_rng(1)
    # Recompute from the same earlier start
    extra_weeks = 200
    first = tuesdays[0] - timedelta(weeks=extra_weeks)
    while first.weekday() != 1:
        first += timedelta(days=1)
    all_t = [first + timedelta(weeks=i) for i in range(n + extra_weeks)]
    vals2 = 1.5 + np.cumsum(rng.normal(0, 0.02, len(all_t)))
    prices["y2"] = pd.DataFrame({"obs_date": all_t, "value": vals2.tolist(), "known_at": all_t})

    # Full-run features
    df_full = build(prices, spec, tuesdays)

    # Check a sample of weeks, spaced across the series
    check_weeks = tuesdays[150::50]  # indices 150, 200, 250, ...
    for t in check_weeks:
        # Truncate prices to known_at <= t
        masked: dict[str, pd.DataFrame] = {}
        for iid, df in prices.items():
            masked[iid] = df[df["known_at"] <= t].copy()

        df_asof = build(masked, spec, [t])
        row_full = df_full.loc[pd.Timestamp(t)]
        row_asof = df_asof.iloc[0]

        for col in df_full.columns:
            vf = row_full[col]
            va = row_asof[col]
            if math.isnan(vf):
                assert math.isnan(va), f"col={col}, t={t}: full=NaN but asof={va}"
            else:
                assert va == pytest.approx(vf, rel=1e-7, abs=1e-12), (
                    f"Look-ahead detected: col={col}, t={t}, full={vf}, asof={va}"
                )


# ── SPEC_V1 sanity check ──────────────────────────────────────────────────────


def test_spec_v1_column_names_are_unique():
    cols = [f.col for f in SPEC_V1]
    assert len(cols) == len(set(cols))


def test_spec_v1_has_expected_feature_count():
    assert len(SPEC_V1) == 51


def test_spec_v1_transforms_are_valid():
    valid = {"log_return", "yield_change", "drawdown", "rvol", "z_score", "spread"}
    for feat in SPEC_V1:
        assert feat.transform in valid, f"{feat.col} has unknown transform {feat.transform!r}"


def test_spec_v1_spread_features_have_instrument_id_2():
    for feat in SPEC_V1:
        if feat.transform == "spread":
            assert feat.instrument_id_2, f"{feat.col}: spread must have instrument_id_2"


def test_spec_v1_no_bloomberg_ews_instruments():
    ews_instruments = {"baltic_dry", "de_30y_yield", "usd_libor_1m"}  # bloomberg-only
    for feat in SPEC_V1:
        assert feat.instrument_id not in ews_instruments
        assert feat.instrument_id_2 not in ews_instruments


def test_spec_v1_build_runs_on_synthetic_data():
    """SPEC_V1 produces a frame with 51 columns without errors on synthetic data."""
    tuesdays = _tuesdays(date(2005, 1, 4), 300)
    prices = _make_prices_for_spec(SPEC_V1, tuesdays, extra_weeks=200)

    df = build(prices, SPEC_V1, tuesdays)

    assert df.shape == (300, 51)
    assert set(df.columns) == {f.col for f in SPEC_V1}
    # After warm-up, most features should be non-NaN
    late = df.iloc[220:]
    nan_frac = late.isna().mean().mean()
    assert nan_frac < 0.05, f"Too many NaNs in late sample: {nan_frac:.1%}"


# ── Point-in-time correctness: rvol known_at ─────────────────────────────────


def test_rvol_holiday_no_look_ahead():
    """rvol at a Tuesday equals rvol from masked data when the next day is a holiday.

    With calendar-filling (resample+ffill) the holiday Wednesday gets the Tuesday
    price, producing a zero log-return that's stamped known_at=Tuesday.  The full
    series samples this Wednesday entry (higher obs_date, same known_at) while the
    masked series stops at Tuesday, giving different values.  With actual-observations-
    only, both full and masked data produce the same result.
    """
    # ~32 business days of history through Tue Jan 14; Wed Jan 15 is the holiday;
    # Thu Jan 16 appears only in the full series.
    all_bdays = [d.date() for d in pd.bdate_range(date(2019, 12, 1), date(2020, 1, 16))]
    obs_dates = [d for d in all_bdays if d != date(2020, 1, 15)]  # drop the holiday

    rng = np.random.default_rng(42)
    prices = np.cumprod(1 + rng.normal(0, 0.01, len(obs_dates))) * 100.0
    df = pd.DataFrame({"obs_date": obs_dates, "value": prices.tolist(), "known_at": obs_dates})

    target_tuesday = date(2020, 1, 14)
    tues_idx = pd.DatetimeIndex([pd.Timestamp(target_tuesday)])

    rv_full = _rvol(df, tues_idx, 20)

    masked = df[df["known_at"] <= target_tuesday].copy()
    rv_masked = _rvol(masked, tues_idx, 20)

    vf = rv_full.iloc[0]
    vm = rv_masked.iloc[0]
    assert not math.isnan(vf), "Expected a valid rvol at the target Tuesday"
    assert vm == pytest.approx(vf, rel=1e-9), f"Holiday look-ahead: full={vf}, masked={vm}"


def test_rvol_respects_known_at_lag():
    """rvol with publication_lag_days=10 equals rvol from data truncated to known_at <= t."""
    n_days = 200
    lag = 10
    d_start = date(2020, 1, 2)
    rng = np.random.default_rng(77)
    rets = rng.normal(0, 0.01, n_days)
    prices = np.cumprod(1 + rets) * 100.0

    idx = pd.bdate_range(d_start, periods=n_days)
    daily_df = pd.DataFrame(
        {
            "obs_date": [d.date() for d in idx],
            "value": prices.tolist(),
            "known_at": [(d + timedelta(days=lag)).date() for d in idx],
        }
    )

    tuesdays = _tuesdays(d_start + timedelta(days=120), 10)
    tues_idx = pd.DatetimeIndex([pd.Timestamp(t) for t in tuesdays])

    rv_full = _rvol(daily_df, tues_idx, 20)

    for t in tuesdays:
        masked = daily_df[daily_df["known_at"] <= t].copy()
        rv_masked = _rvol(masked, pd.DatetimeIndex([pd.Timestamp(t)]), 20)
        vf = rv_full.loc[pd.Timestamp(t)]
        vm = rv_masked.iloc[0]
        if math.isnan(vf):
            assert math.isnan(vm), f"t={t}: full=NaN but masked={vm}"
        else:
            assert vm == pytest.approx(vf, rel=1e-9), (
                f"rvol point-in-time violation at t={t}: full={vf}, masked={vm}"
            )


# ── Stale carry-forward ───────────────────────────────────────────────────────


def test_weekly_series_monthly_gap_stale_after_45_days():
    """A monthly series returns NaN for Tuesdays more than 45 days after last known_at."""
    # Six monthly observations; no data after June 1
    monthly_dates = [date(2020, m, 1) for m in range(1, 7)]
    df = _price_df(monthly_dates, [100.0] * 6, lag_days=0)

    # A Tuesday strictly beyond 45 days after Jun 1 (the last known_at)
    beyond = date(2020, 6, 1) + timedelta(days=46)
    while beyond.weekday() != 1:
        beyond += timedelta(days=1)

    tues_idx = pd.DatetimeIndex([pd.Timestamp(beyond)])
    s = _weekly_series(df, tues_idx)
    assert math.isnan(s.iloc[0])


def test_weekly_series_gap_returns_nan():
    """A daily series with a 1-year gap produces NaN for weeks inside the gap."""
    before = pd.bdate_range(date(2019, 1, 2), periods=65)
    after = pd.bdate_range(date(2020, 4, 1), periods=10)
    all_obs = [d.date() for d in before] + [d.date() for d in after]
    df = _price_df(all_obs, [100.0] * len(all_obs), lag_days=0)

    # Tuesday well inside the gap (Dec 2019 — ~9 months after data stops)
    gap_tuesday = date(2019, 12, 3)
    tues_idx = pd.DatetimeIndex([pd.Timestamp(gap_tuesday)])
    s = _weekly_series(df, tues_idx)
    assert math.isnan(s.iloc[0])


# ── Provider splicing ─────────────────────────────────────────────────────────


from market_anomaly_detector.features.build import _splice_providers, load_prices  # noqa: E402


def _provider_df(start: date, n_days: int, val_start: float = 100.0) -> pd.DataFrame:
    """Daily DataFrame with n_days consecutive calendar days starting on start."""
    obs = [start + timedelta(days=i) for i in range(n_days)]
    return pd.DataFrame(
        {
            "obs_date": obs,
            "value": [val_start + i * 0.01 for i in range(n_days)],
            "known_at": obs,
        }
    )


def test_splice_providers_returns_none_for_empty():
    assert _splice_providers([]) is None


def test_splice_providers_single_provider_adds_source_column():
    df = _provider_df(date(2020, 1, 1), 30)
    result = _splice_providers([("yahoo", "proxy", df)])
    assert result is not None
    assert "source_provider" in result.columns
    assert (result["source_provider"] == "yahoo").all()
    assert len(result) == 30


def test_splice_providers_extends_backward_no_overlap():
    """Secondary rows strictly before primary start are prepended; overlap excluded."""
    primary = _provider_df(date(2020, 1, 10), 11)  # days 10–20
    secondary = _provider_df(date(2020, 1, 1), 15)  # days 1–15 (overlap 10–15)
    result = _splice_providers([("yahoo", "exact", primary), ("fred", "proxy", secondary)])
    assert result is not None
    # Secondary contributes days 1–9 (9 rows), primary keeps days 10–20 (11 rows)
    assert len(result) == 9 + 11
    sec = result[result["source_provider"] == "fred"]
    pri = result[result["source_provider"] == "yahoo"]
    assert len(sec) == 9
    assert len(pri) == 11
    assert sec["obs_date"].max() < pri["obs_date"].min()


def test_splice_providers_skips_low_fidelity_secondary():
    """Secondary with fidelity not in {proxy, exact} is never used."""
    primary = _provider_df(date(2020, 1, 10), 11)
    secondary = _provider_df(date(2020, 1, 1), 15)
    result = _splice_providers([("yahoo", "exact", primary), ("fred", "unavailable", secondary)])
    assert result is not None
    assert (result["source_provider"] == "yahoo").all()
    assert len(result) == 11


def test_splice_providers_chain_two_secondaries():
    """Second secondary extends before the first secondary (chain splice)."""
    primary = _provider_df(date(2020, 6, 1), 30)  # starts Jun 1
    sec1 = _provider_df(date(2020, 3, 1), 100)  # starts Mar 1
    sec2 = _provider_df(date(2020, 1, 1), 200)  # starts Jan 1
    result = _splice_providers(
        [("yahoo", "exact", primary), ("fred", "proxy", sec1), ("cboe", "proxy", sec2)]
    )
    assert result is not None
    min_by_prov = result.groupby("source_provider")["obs_date"].min()
    # cboe earliest, fred middle, yahoo latest
    assert min_by_prov["cboe"] < min_by_prov["fred"] < min_by_prov["yahoo"]


def test_splice_providers_noop_when_secondary_does_not_pre_date_primary():
    """Secondary starting after primary is not spliced; primary unchanged."""
    primary = _provider_df(date(2020, 1, 1), 30)
    secondary = _provider_df(date(2020, 1, 15), 30)  # starts after primary
    result = _splice_providers([("yahoo", "exact", primary), ("fred", "proxy", secondary)])
    assert result is not None
    assert (result["source_provider"] == "yahoo").all()
    assert len(result) == 30


def test_splice_providers_sorted_by_obs_date():
    """Result rows are in ascending obs_date order after splice."""
    primary = _provider_df(date(2020, 6, 1), 10)
    secondary = _provider_df(date(2020, 1, 1), 50)
    result = _splice_providers([("yahoo", "proxy", primary), ("fred", "proxy", secondary)])
    assert result is not None
    dates = result["obs_date"].tolist()
    assert dates == sorted(dates)


# ── FeatureSpec.fallback_instrument_ids ───────────────────────────────────────


def test_featurespec_fallback_instrument_ids_defaults_to_empty():
    """FeatureSpec.fallback_instrument_ids defaults to an empty list."""
    feat = FeatureSpec("x_lr_1w", "X 1w", "x", "price", "log_return", 1)
    assert feat.fallback_instrument_ids == []


def test_featurespec_fallback_instrument_ids_accepted():
    """FeatureSpec can be created with explicit fallback_instrument_ids."""
    feat = FeatureSpec(
        "x_lr_1w",
        "X 1w",
        "x",
        "price",
        "log_return",
        1,
        fallback_instrument_ids=["x_proxy1", "x_proxy2"],
    )
    assert feat.fallback_instrument_ids == ["x_proxy1", "x_proxy2"]


# ── validate_spec_fallbacks ───────────────────────────────────────────────────


from market_anomaly_detector.features.spec import validate_spec_fallbacks  # noqa: E402


class _FakeInst:
    def __init__(self, id_: str) -> None:
        self.id = id_


def test_validate_spec_fallbacks_passes_for_known_ids():
    """validate_spec_fallbacks does not raise when all fallback IDs are in the registry."""
    instruments = [_FakeInst("main"), _FakeInst("proxy_a")]
    spec = [
        FeatureSpec(
            "main_lr_1w",
            "Main 1w",
            "main",
            "price",
            "log_return",
            1,
            fallback_instrument_ids=["proxy_a"],
        )
    ]
    validate_spec_fallbacks(spec, instruments)  # should not raise


def test_validate_spec_fallbacks_raises_for_unknown_id():
    """validate_spec_fallbacks raises ValueError naming the unknown fallback ID."""
    instruments = [_FakeInst("main")]
    spec = [
        FeatureSpec(
            "main_lr_1w",
            "Main 1w",
            "main",
            "price",
            "log_return",
            1,
            fallback_instrument_ids=["nonexistent_proxy"],
        )
    ]
    with pytest.raises(ValueError, match="nonexistent_proxy"):
        validate_spec_fallbacks(spec, instruments)


def test_validate_spec_fallbacks_passes_for_empty_fallbacks():
    """validate_spec_fallbacks passes when no spec entry has fallback_instrument_ids."""
    instruments = [_FakeInst("main")]
    spec = [FeatureSpec("main_lr_1w", "Main 1w", "main", "price", "log_return", 1)]
    validate_spec_fallbacks(spec, instruments)  # should not raise


# ── load_prices cross-instrument fallback ────────────────────────────────────


class _SRow:
    """Fake SQLAlchemy SymbolMapRow result for mock sessions."""

    def __init__(self, provider: str, priority: int, fidelity: str) -> None:
        self.provider = provider
        self.priority = priority
        self.fidelity = fidelity


class _PRow:
    """Fake SQLAlchemy PriceRow result for mock sessions; iterable so pandas can unpack it."""

    def __init__(self, obs_date: date, value: float, known_at: date) -> None:
        self.obs_date = obs_date
        self.value = value
        self.known_at = known_at

    def __iter__(self):
        return iter([self.obs_date, self.value, self.known_at])

    def __len__(self):
        return 3


def _make_mock_session(
    symbol_rows: dict[str, list[tuple[str, int, str]]],
    price_rows: dict[tuple[str, str], list[tuple[date, float, date]]],
) -> object:
    """Build a mock SQLAlchemy session for load_prices tests.

    Uses a FIFO queue: calls are returned in the order load_prices issues them.
    For each instrument (sorted), load_prices calls:
      1. execute(symbol_map_query) → _SRow list
      2. execute(price_query) × N providers → _PRow list per provider

    symbol_rows: {instrument_id: [(provider, priority, fidelity), ...]}
    price_rows:  {(instrument_id, provider): [(obs_date, value, known_at), ...]}
    """
    from unittest.mock import MagicMock

    # Build call queue in the exact order load_prices issues queries.
    # load_prices calls _fetch_providers for each primary inst (sorted), then
    # for each fallback inst as they appear in fallback_map.
    # We can't predict fallback order here, so instead we intercept by
    # inspecting the compiled SQL params.

    session = MagicMock()

    def execute(stmt):
        result = MagicMock()
        try:
            compiled = stmt.compile()
            params = compiled.params
        except Exception:
            result.fetchall.return_value = []
            return result

        # instrument_id bind param name varies; try common names
        inst_id = (
            params.get("instrument_id_1")
            or params.get("instrument_id_1_1")
            or params.get("instrument_id_1_2")
        )
        provider = params.get("provider_1") or params.get("provider_1_1")

        if inst_id is None:
            result.fetchall.return_value = []
            return result

        if provider is None or provider == "bloomberg_ews":
            # Symbol map query (filters provider != "bloomberg_ews")
            srcs = [s for s in symbol_rows.get(inst_id, []) if s[0] != "bloomberg_ews"]
            result.fetchall.return_value = [_SRow(*s) for s in srcs]
        else:
            # Price query
            key = (inst_id, provider)
            rows = price_rows.get(key, [])
            result.fetchall.return_value = [_PRow(*r) for r in rows]

        return result

    session.execute.side_effect = execute
    return session


def test_load_prices_cross_instrument_splice():
    """load_prices splices fallback instrument data before primary's first date.

    Primary has data from 2020-01-10; fallback from 2020-01-01 to 2020-01-09
    (the two rows before primary start) are spliced in.
    """
    primary_dates = [date(2020, 1, 10), date(2020, 1, 17), date(2020, 1, 24)]
    fallback_dates = [date(2020, 1, 1), date(2020, 1, 5), date(2020, 1, 9)]

    symbol_rows = {
        "main_inst": [("yahoo", 1, "proxy")],
        "fallback_inst": [("yahoo", 1, "proxy")],
    }
    price_rows = {
        ("main_inst", "yahoo"): [(d, float(100 + i), d) for i, d in enumerate(primary_dates)],
        ("fallback_inst", "yahoo"): [(d, float(90 + i), d) for i, d in enumerate(fallback_dates)],
    }

    spec = [
        FeatureSpec(
            "x_lr_1w",
            "X 1w",
            "main_inst",
            "price",
            "log_return",
            1,
            fallback_instrument_ids=["fallback_inst"],
        )
    ]

    session = _make_mock_session(symbol_rows, price_rows)
    result = load_prices(session, spec)

    assert "main_inst" in result
    df = result["main_inst"]
    # Should have primary rows + fallback rows before primary start
    assert len(df) == len(primary_dates) + len(fallback_dates)
    # All fallback rows have earlier dates than any primary row
    fb_rows = df[df["source_instrument_id"] == "fallback_inst"]
    pri_rows = df[df["source_instrument_id"] == "main_inst"]
    assert not fb_rows.empty
    assert not pri_rows.empty
    assert fb_rows["obs_date"].max() < pri_rows["obs_date"].min()
    # source_instrument_id column exists on all rows
    assert "source_instrument_id" in df.columns


def test_load_prices_skips_low_fidelity_fallback():
    """Fallback instruments rated 'unavailable' are never spliced in."""
    primary_dates = [date(2020, 1, 10)]
    fallback_dates = [date(2020, 1, 1)]

    symbol_rows = {
        "main_inst": [("yahoo", 1, "exact")],
        "bad_fallback": [("yahoo", 1, "unavailable")],
    }
    price_rows = {
        ("main_inst", "yahoo"): [(d, 100.0, d) for d in primary_dates],
        ("bad_fallback", "yahoo"): [(d, 80.0, d) for d in fallback_dates],
    }

    spec = [
        FeatureSpec(
            "x_lr_1w",
            "X 1w",
            "main_inst",
            "price",
            "log_return",
            1,
            fallback_instrument_ids=["bad_fallback"],
        )
    ]

    session = _make_mock_session(symbol_rows, price_rows)
    result = load_prices(session, spec)

    assert "main_inst" in result
    df = result["main_inst"]
    # Only primary rows; fallback skipped due to unavailable fidelity
    assert len(df) == len(primary_dates)
    assert (df["source_instrument_id"] == "main_inst").all()


def test_load_prices_source_instrument_id_set_without_fallbacks():
    """source_instrument_id is always populated, even when no fallbacks are used."""
    primary_dates = [date(2020, 1, 7), date(2020, 1, 14)]
    symbol_rows = {"my_inst": [("yahoo", 1, "exact")]}
    price_rows = {("my_inst", "yahoo"): [(d, 100.0, d) for d in primary_dates]}

    spec = [FeatureSpec("x_lr_1w", "X 1w", "my_inst", "price", "log_return", 1)]
    session = _make_mock_session(symbol_rows, price_rows)
    result = load_prices(session, spec)

    assert "my_inst" in result
    df = result["my_inst"]
    assert "source_instrument_id" in df.columns
    assert (df["source_instrument_id"] == "my_inst").all()


# ── spec_hash ────────────────────────────────────────────────────────────────


def test_spec_hash_v1_pinned():
    """spec_hash(SPEC_V1) must equal the value measured at commit 2dfa940.

    Any edit to SPEC_V1 will change this hash and fail CI, which is the intent.
    The hash covers col, instrument_id, instrument_id_2, transform, window,
    and fallback_instrument_ids; it excludes label and units so cosmetic renames
    don't trip the guard.
    """
    assert spec_hash(SPEC_V1) == "b8c2d65c449392a806bf30c8fc596785cab943aa08268fae4fff593b14ab8a3f"


def test_spec_hash_v2_differs_from_v1():
    """SPEC_V2 adds fallbacks to GBP/USD entries, so its hash must differ from V1."""
    assert spec_hash(SPEC_V2) != spec_hash(SPEC_V1)


def test_spec_hash_stable_across_calls():
    """spec_hash is deterministic — identical inputs always produce the same digest."""
    assert spec_hash(SPEC_V1) == spec_hash(SPEC_V1)


def test_spec_hash_changes_on_col_edit():
    """Renaming a col must change the hash (sanity-check the hash logic)."""
    modified = [
        FeatureSpec(
            "vix_lr_1w_RENAMED" if f.col == "vix_lr_1w" else f.col,
            f.label,
            f.instrument_id,
            f.units,
            f.transform,
            f.window,
            f.instrument_id_2,
            f.fallback_instrument_ids,
        )
        for f in SPEC_V1
    ]
    assert spec_hash(modified) != spec_hash(SPEC_V1)
