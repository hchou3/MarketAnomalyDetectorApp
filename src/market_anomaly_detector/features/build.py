"""Feature builder for the Market Anomaly Detector.

Core entry points
-----------------
build(prices, spec, tuesdays, as_of=None) -> pd.DataFrame
    Pure function: compute features from a pre-fetched prices dict.
    Tests call this directly with synthetic data.

load_prices(session, spec, as_of=None) -> dict[str, pd.DataFrame]
    Fetch prices for every instrument in spec from the database.
    Picks the highest-priority non-bloomberg provider that has data.

build_from_db(session, spec, tuesdays, as_of=None) -> pd.DataFrame
    Convenience wrapper: load_prices → build.

Point-in-time guarantee
-----------------------
For each Tuesday t, an observation is usable only when known_at <= t.
Weekly alignment uses merge_asof on the known_at column with a staleness
tolerance inferred from the data's native frequency and publication lag.
Realized-vol daily series is stamped by known_at before the Tuesday
merge_asof, so publication-lagged series are handled correctly.
"""

from __future__ import annotations

import math
from datetime import date
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .spec import FeatureSpec

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

# z-score rolling minimum: require at least this many weeks before producing a value
_Z_MIN_PERIODS = 104

# maximum gap (days) between a Tuesday and the most recent daily rvol known_at
_RVOL_STALE_DAYS = 7

# extra historical tuesdays prepended before the requested range so that rolling
# transforms (z_score window=156, log_return window=12, drawdown window=52) have
# sufficient context even when build() is called for a single Tuesday
_MAX_LOOKBACK_WEEKS = 160


def build(
    prices: dict[str, pd.DataFrame],
    spec: list[FeatureSpec],
    tuesdays: list[date],
    as_of: date | None = None,
) -> pd.DataFrame:
    """Compute features for each Tuesday in *tuesdays*.

    Parameters
    ----------
    prices:
        ``{instrument_id: DataFrame}`` where each DataFrame has columns
        ``obs_date`` (date), ``value`` (float), ``known_at`` (date).
    spec:
        List of :class:`FeatureSpec` defining which features to compute.
    tuesdays:
        Ordered list of weekly anchor dates (W-TUE).
    as_of:
        If given, mask all prices to ``known_at <= as_of`` before building.
        Use for single-week live scoring.

    Returns
    -------
    pd.DataFrame
        Float frame indexed by ``pd.DatetimeIndex`` of the Tuesday dates;
        columns match ``feat.col`` for each spec entry.
        Missing values propagate as ``float("nan")``.
    """
    tues_idx = pd.DatetimeIndex([pd.Timestamp(t) for t in tuesdays])

    # Mask prices to known_at <= as_of when scoring a single point in time
    effective: dict[str, pd.DataFrame] = {}
    for inst_id, df in prices.items():
        if as_of is not None:
            df = df[df["known_at"] <= as_of].copy()
        effective[inst_id] = df

    # Extend the tuesday index backward so rolling transforms (z_score, drawdown,
    # multi-week returns) have enough historical context even for single-week calls.
    # Results for the extra rows are discarded at the end.
    if len(tues_idx) > 0:
        extra = pd.date_range(
            end=tues_idx[0] - pd.Timedelta(weeks=1),
            periods=_MAX_LOOKBACK_WEEKS,
            freq="W-TUE",
        )
        extended_idx = extra.append(tues_idx)
    else:
        extended_idx = tues_idx

    # Pre-build weekly series (shared across features for the same instrument)
    weekly: dict[str, pd.Series] = {}
    for feat in spec:
        for iid in (feat.instrument_id, feat.instrument_id_2):
            if iid and iid not in weekly:
                df = effective.get(iid)
                weekly[iid] = (
                    _weekly_series(df, extended_idx)
                    if df is not None
                    else _nan_series(extended_idx)
                )

    # Raw DataFrames for rvol features — _rvol handles known_at internally
    rvol_dfs: dict[str, pd.DataFrame | None] = {}
    for feat in spec:
        if feat.transform == "rvol" and feat.instrument_id not in rvol_dfs:
            rvol_dfs[feat.instrument_id] = effective.get(feat.instrument_id)

    # Compute each feature on the extended index
    out_ext = pd.DataFrame(index=extended_idx, dtype=float)
    for feat in spec:
        t = feat.transform
        s = weekly.get(feat.instrument_id, _nan_series(extended_idx))
        if t == "log_return":
            out_ext[feat.col] = _log_return(s, feat.window)
        elif t == "yield_change":
            out_ext[feat.col] = _yield_change(s, feat.window)
        elif t == "drawdown":
            out_ext[feat.col] = _drawdown(s, feat.window)
        elif t == "rvol":
            out_ext[feat.col] = _rvol(rvol_dfs.get(feat.instrument_id), extended_idx, feat.window)
        elif t == "z_score":
            out_ext[feat.col] = _z_score(s, feat.units)
        elif t == "spread":
            s2 = weekly.get(feat.instrument_id_2, _nan_series(extended_idx))
            out_ext[feat.col] = _spread(s, s2)
        else:
            raise ValueError(f"Unknown transform: {feat.transform!r}")

    # Slice back to the originally requested tuesdays
    out = out_ext.loc[tues_idx].copy()

    # Rows after as_of are undefined (as_of scoring is only valid up to as_of)
    if as_of is not None:
        out.loc[out.index > pd.Timestamp(as_of)] = float("nan")

    return out


def load_prices(
    session: Session,
    spec: list[FeatureSpec],
    as_of: date | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch prices for every instrument referenced by *spec* from the database.

    For each instrument the provider with the lowest priority number (highest
    priority) that actually has data is used.  ``bloomberg_ews`` is never
    selected.  If *as_of* is given, rows with ``known_at > as_of`` are excluded.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are instrument IDs.  Each DataFrame has float ``value``,
        date ``obs_date``, and date ``known_at``.
    """
    from sqlalchemy import select

    from ..db.models import PriceRow, SymbolMapRow

    instrument_ids: set[str] = set()
    for feat in spec:
        instrument_ids.add(feat.instrument_id)
        if feat.instrument_id_2:
            instrument_ids.add(feat.instrument_id_2)

    result: dict[str, pd.DataFrame] = {}
    for inst_id in sorted(instrument_ids):
        providers = session.execute(
            select(SymbolMapRow.provider, SymbolMapRow.priority)
            .where(
                SymbolMapRow.instrument_id == inst_id,
                SymbolMapRow.provider != "bloomberg_ews",
            )
            .order_by(SymbolMapRow.priority)
        ).fetchall()

        for prow in providers:
            q = select(PriceRow.obs_date, PriceRow.value, PriceRow.known_at).where(
                PriceRow.instrument_id == inst_id,
                PriceRow.provider == prow.provider,
            )
            if as_of is not None:
                q = q.where(PriceRow.known_at <= as_of)
            rows = session.execute(q.order_by(PriceRow.obs_date)).fetchall()
            if rows:
                df = pd.DataFrame(rows, columns=["obs_date", "value", "known_at"])
                df["value"] = df["value"].astype(float)
                result[inst_id] = df
                break  # highest-priority provider with data wins

    return result


def build_from_db(
    session: Session,
    spec: list[FeatureSpec],
    tuesdays: list[date],
    as_of: date | None = None,
) -> pd.DataFrame:
    """Fetch prices from the DB and compute features in one call."""
    prices = load_prices(session, spec, as_of=as_of)
    return build(prices, spec, tuesdays, as_of=as_of)


# ── Internal helpers ──────────────────────────────────────────────────────────


def _nan_series(idx: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(float("nan"), index=idx, dtype=float)


def _stale_tolerance(df: pd.DataFrame) -> pd.Timedelta:
    """Infer carry-forward tolerance from the data's native observation frequency.

    Frequency bucket → tolerance: daily (min gap ≤ 2d) → 7d, weekly (≤ 10d) → 14d,
    monthly → 45d.  The merge is already on known_at, which encodes the
    publication lag, so lag must not be added to the tolerance again.
    """
    if df is None or df.empty:
        return pd.Timedelta(days=45)
    obs_sorted = pd.to_datetime(df["obs_date"]).sort_values()
    gaps = obs_sorted.diff().dt.days.dropna()
    pos = gaps[gaps > 0]
    min_gap = int(pos.min()) if len(pos) > 0 else 1
    buffer = 7 if min_gap <= 2 else (14 if min_gap <= 10 else 45)
    return pd.Timedelta(days=buffer)


def _weekly_series(df: pd.DataFrame, tues_idx: pd.DatetimeIndex) -> pd.Series:
    """For each Tuesday, return the most recently published value (known_at <= T).

    Values are carried forward only within the tolerance inferred from the
    data's native frequency and publication lag.  Gaps beyond the tolerance
    return NaN instead of a stale value.

    Sorting on (known_at, obs_date) ensures that when two rows share the same
    known_at the more recent observation wins.
    """
    if df is None or df.empty:
        return _nan_series(tues_idx)

    df = df.copy()
    df["_ka_dt"] = pd.to_datetime(df["known_at"])
    df = df.sort_values(["_ka_dt", "obs_date"], ascending=[True, True]).reset_index(drop=True)

    tues_df = pd.DataFrame({"_tdt": tues_idx})
    merged = pd.merge_asof(
        tues_df,
        df[["_ka_dt", "value"]],
        left_on="_tdt",
        right_on="_ka_dt",
        direction="backward",
        tolerance=_stale_tolerance(df),
    )
    return pd.Series(merged["value"].values, index=tues_idx, dtype=float)


def _log_return(s: pd.Series, weeks: int) -> pd.Series:
    """log(P[t]) - log(P[t-w])."""
    log_s = np.log(s.astype(float))
    return log_s - log_s.shift(weeks)


def _yield_change(s: pd.Series, weeks: int) -> pd.Series:
    """(P[t] - P[t-w]) * 100  in basis points."""
    return (s - s.shift(weeks)) * 100.0


def _drawdown(s: pd.Series, weeks: int) -> pd.Series:
    """log(P[t]) - max(log(P[t-w+1:t+1]))  (trailing window including current week)."""
    log_s = np.log(s.astype(float))
    rolling_max = log_s.rolling(weeks, min_periods=1).max()
    return log_s - rolling_max


def _rvol(df: pd.DataFrame | None, tues_idx: pd.DatetimeIndex, window_days: int) -> pd.Series:
    """Annualised realised vol of daily log returns, sampled at each Tuesday.

    Respects known_at: for each Tuesday t, only prices with known_at <= t
    contribute.  Log returns are computed between consecutive actual
    observations only (no calendar filling); each return is stamped with
    the later observation's known_at.
    """
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return _nan_series(tues_idx)

    work = df.copy()
    work["_obs_dt"] = pd.to_datetime(work["obs_date"])
    work["_ka_dt"] = pd.to_datetime(work["known_at"])
    # Dedup: if multiple rows share obs_date, keep the one with the latest known_at
    work = (
        work.sort_values(["_obs_dt", "_ka_dt"])
        .groupby("_obs_dt", as_index=False)
        .last()
        .sort_values("_obs_dt")
        .reset_index(drop=True)
    )

    # Log returns between consecutive actual observations (no calendar filling).
    # rv_series.iloc[i] uses work["_ka_dt"].iloc[i] as its publication date.
    log_ret = np.log(work["value"].astype(float)).diff()
    rv_series = log_ret.rolling(
        window_days, min_periods=max(1, window_days // 2)
    ).std() * math.sqrt(252)

    rv_with_ka = (
        pd.DataFrame(
            {
                "_obs_dt": work["_obs_dt"].values,
                "_ka_dt": work["_ka_dt"].values,
                "_rv": rv_series.values,
            }
        )
        .dropna(subset=["_rv"])
        .sort_values(["_ka_dt", "_obs_dt"])
    )

    # Sample at each Tuesday: latest rvol where known_at <= tuesday
    tues_df = pd.DataFrame({"_tdt": tues_idx})
    merged = pd.merge_asof(
        tues_df,
        rv_with_ka[["_ka_dt", "_rv"]],
        left_on="_tdt",
        right_on="_ka_dt",
        direction="backward",
        tolerance=pd.Timedelta(days=_RVOL_STALE_DAYS),
    )
    return pd.Series(merged["_rv"].values, index=tues_idx, dtype=float)


def _z_score(s: pd.Series, units: str) -> pd.Series:
    """Z-score of the 1-week return/change over a trailing 156-week window.

    For price/index_points: 1-week log return.
    For percent: 1-week yield change in basis points.
    Min periods: _Z_MIN_PERIODS (104 weeks ≈ 2 years).
    """
    if units in ("price", "index_points"):
        ret = _log_return(s, 1)
    else:
        ret = _yield_change(s, 1)

    z_window = 156
    roll_mean = ret.rolling(z_window, min_periods=_Z_MIN_PERIODS).mean()
    roll_std = ret.rolling(z_window, min_periods=_Z_MIN_PERIODS).std()
    return (ret - roll_mean) / roll_std


def _spread(s1: pd.Series, s2: pd.Series) -> pd.Series:
    """(P1[t] - P2[t]) * 100  in basis points."""
    return (s1 - s2) * 100.0
