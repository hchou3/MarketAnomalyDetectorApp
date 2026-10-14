"""Yahoo Finance provider via yfinance.

``auto_adjust`` is specified per-symbol via the ``symbol_auto_adjust`` dict
passed to the constructor.  ETF proxies should use ``True`` (splits and
dividends folded into the Close price) so total-return comparisons are valid.
Indices, futures, and FX pairs should use ``False`` (raw unadjusted Close).

Dates are kept in exchange-local time: the index timezone is stripped and
the timestamp is normalised to midnight so obs_date matches the exchange
calendar, not UTC.  This mirrors the convention used in fetch_stress_data.py.
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from typing import Callable, Optional

import pandas as pd


def _yfinance_fetch(
    symbol: str,
    start: Optional[date],
    end: Optional[date],
    auto_adjust: bool,
) -> pd.DataFrame:
    import yfinance as yf

    kwargs: dict = {"auto_adjust": auto_adjust, "progress": False}
    if start is not None:
        kwargs["start"] = start.isoformat()
        if end is not None:
            # yfinance end is exclusive; add 1 day so the requested end date is included
            kwargs["end"] = (end + timedelta(days=1)).isoformat()
    else:
        kwargs["period"] = "max"

    return yf.Ticker(symbol).history(**kwargs)


class YahooProvider:
    """Fetch price history from Yahoo Finance via yfinance."""

    name = "yahoo"

    def __init__(
        self,
        symbol_auto_adjust: Optional[dict[str, bool]] = None,
        *,
        request_delay: float = 0.5,
        _fetcher: Optional[Callable] = None,
    ) -> None:
        self._symbol_auto_adjust: dict[str, bool] = symbol_auto_adjust or {}
        # Skip the polite delay when a test fetcher is injected.
        self._request_delay = request_delay if _fetcher is None else 0.0
        self._fetch = _fetcher if _fetcher is not None else _yfinance_fetch

    def get_history(
        self,
        symbol: str,
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> pd.DataFrame:
        """Return DataFrame[obs_date: date, value: float].

        Exchange-local dates are used (timezone stripped, normalised to
        midnight).  An empty or all-NaN result raises ValueError.
        """
        auto_adjust = self._symbol_auto_adjust.get(symbol, False)
        raw = self._fetch(symbol, start, end, auto_adjust)

        if raw.empty:
            raise ValueError(f"Yahoo Finance returned no data for {symbol!r}")

        # Strip timezone and normalise to exchange-local midnight
        idx = raw.index
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        idx = idx.normalize()

        close = raw["Close"].copy()
        close.index = idx

        result = pd.DataFrame(
            {
                "obs_date": close.index.date,
                "value": pd.to_numeric(close.values, errors="coerce"),
            }
        )
        result = result.dropna(subset=["value"]).copy()

        if result.empty:
            raise ValueError(f"Yahoo Finance {symbol!r}: no valid Close values after cleaning")

        if start is not None:
            result = result[result["obs_date"] >= start]
        if end is not None:
            result = result[result["obs_date"] <= end]

        if self._request_delay:
            time.sleep(self._request_delay)

        return result.sort_values("obs_date").reset_index(drop=True)
