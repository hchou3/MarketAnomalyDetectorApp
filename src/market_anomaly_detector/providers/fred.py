"""FRED provider — fetches time-series from the St. Louis Fed.

Uses the public fredgraph.csv endpoint; no API key required for current-vintage
data.  Date range is sent as ``cosd`` / ``coed`` query parameters so only the
requested window is transferred.

If ``FRED_API_KEY`` is set in the environment, ``scripts/record_fixtures.py``
can additionally query ALFRED for vintage history to measure publication lags.
"""

from __future__ import annotations

import io
from datetime import date
from typing import Callable, Optional
from urllib.parse import urlencode

import pandas as pd

from ._http import http_get

_FREDGRAPH_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def _strip_preamble(text: str) -> str:
    """Return text starting from the first line whose first token is 'DATE'.

    FRED's fredgraph.csv endpoint sometimes prepends metadata lines (Title,
    Series ID, Source, Release) before the actual DATE,VALUE header row.
    This function skips those lines so pd.read_csv always sees a clean CSV.
    Old-format responses (no preamble) pass through unchanged.
    """
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        token = line.strip().split(",")[0].strip().strip('"').upper()
        if token == "DATE":
            return "".join(lines[i:])
    return text


class FredProvider:
    """Fetch data from FRED using the public fredgraph.csv endpoint.

    Missing values (".") are silently dropped.  An empty or all-missing series
    raises ValueError so callers can fall back to a lower-priority source.
    """

    name = "fred"

    def __init__(self, *, _fetcher: Optional[Callable[[str], str]] = None) -> None:
        self._fetch: Callable[[str], str] = _fetcher if _fetcher is not None else http_get

    def get_history(
        self,
        symbol: str,
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> pd.DataFrame:
        """Return DataFrame[obs_date: date, value: float] for one FRED series.

        ``start`` and ``end`` (inclusive) are passed as ``cosd`` / ``coed``
        query parameters to limit the response to the requested window.
        """
        params: dict[str, str] = {"id": symbol}
        if start is not None:
            params["cosd"] = start.isoformat()
        if end is not None:
            params["coed"] = end.isoformat()
        url = f"{_FREDGRAPH_URL}?{urlencode(params)}"

        text = self._fetch(url)
        df = pd.read_csv(io.StringIO(_strip_preamble(text)))
        if df.shape[1] < 2:
            raise ValueError(f"Unexpected FRED response for {symbol!r}: {text[:200]!r}")

        df.columns = ["obs_date", "value"]
        df["obs_date"] = pd.to_datetime(df["obs_date"]).dt.date
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"]).copy()

        if df.empty:
            raise ValueError(f"FRED series {symbol!r} returned no usable data rows")

        return df.sort_values("obs_date").reset_index(drop=True)
