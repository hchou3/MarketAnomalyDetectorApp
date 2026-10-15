"""Cboe provider — fetches daily index history from Cboe's public CDN.

Two CDN hostnames are tried in order; the second is the fallback.  Date range
filtering is done in Python after the full history is fetched (the CDN has no
query-parameter filtering).
"""

from __future__ import annotations

import io
from datetime import date
from typing import Callable, Optional

import pandas as pd

from ._http import http_get

_CDN_URLS = [
    "https://cdn.cboe.com/api/global/us_indices/daily_prices/{symbol}_History.csv",
    "https://cdn-api.cboe.com/api/global/us_indices/daily_prices/{symbol}_History.csv",
]


class CboeProvider:
    """Fetch daily close history from Cboe's public CDN.

    The primary CDN URL is tried first; if it raises any exception the
    secondary URL is tried.  If both fail, a RuntimeError is raised with
    both error messages.
    """

    name = "cboe"

    def __init__(self, *, _fetcher: Optional[Callable[[str], str]] = None) -> None:
        self._fetch: Callable[[str], str] = _fetcher if _fetcher is not None else http_get

    def get_history(
        self,
        symbol: str,
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> pd.DataFrame:
        """Return DataFrame[obs_date: date, value: float] for a Cboe index."""
        errors: list[str] = []
        text: Optional[str] = None
        for template in _CDN_URLS:
            url = template.format(symbol=symbol)
            try:
                text = self._fetch(url)
                break
            except Exception as exc:
                errors.append(f"{url}: {exc}")

        if text is None:
            raise RuntimeError(f"All Cboe CDN URLs failed for {symbol!r}: " + "; ".join(errors))

        df = pd.read_csv(io.StringIO(text))
        df.columns = [c.strip().upper() for c in df.columns]

        date_col = df.columns[0]
        close_col = "CLOSE" if "CLOSE" in df.columns else df.columns[-1]

        result = pd.DataFrame(
            {
                "obs_date": pd.to_datetime(df[date_col]).dt.date,
                "value": pd.to_numeric(df[close_col], errors="coerce"),
            }
        )
        result = result.dropna(subset=["value"]).copy()

        if result.empty:
            raise ValueError(f"Cboe symbol {symbol!r} returned no data")

        if start is not None:
            result = result[result["obs_date"] >= start]
        if end is not None:
            result = result[result["obs_date"] <= end]

        return result.sort_values("obs_date").reset_index(drop=True)
