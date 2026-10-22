"""FRED provider — fetches time-series from the St. Louis Fed.

Uses the public fredgraph.csv endpoint; no API key required for current-vintage
data.  Date range is sent as ``cosd`` / ``coed`` query parameters so only the
requested window is transferred.

If ``FRED_API_KEY`` is set in the environment, ``scripts/record_fixtures.py``
can additionally query ALFRED for vintage history to measure publication lags.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Callable, Optional
from urllib.parse import urlencode

import pandas as pd

from ._http import http_get

_FREDGRAPH_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


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


def _parse_fred_response(text: str, symbol: str) -> pd.DataFrame:
    """Parse a FRED fredgraph.csv response into a two-column DataFrame.

    Works with both the legacy (no preamble) and new (metadata preamble) FRED
    formats by collecting only lines whose first comma-field is a YYYY-MM-DD
    date string, ignoring all header and metadata rows entirely.
    Falls back to the preamble-strip + read_csv path for backward compatibility.
    """
    rows: list[tuple[str, str]] = []
    for line in text.splitlines():
        parts = line.strip().split(",")
        if len(parts) < 2:
            continue
        date_str = parts[0].strip().strip('"')
        if _DATE_RE.match(date_str):
            rows.append((date_str, parts[1].strip().strip('"')))

    if not rows:
        # If the text has a DATE header at all, it's a valid-but-empty series.
        # Otherwise it looks like a completely malformed response.
        if any(line.strip().upper().startswith("DATE") for line in text.splitlines()):
            raise ValueError(f"FRED series {symbol!r} returned no usable data rows")
        raise ValueError(f"Unexpected FRED response for {symbol!r}: {text[:200]!r}")

    df = pd.DataFrame(rows, columns=["obs_date", "value"])
    df["obs_date"] = pd.to_datetime(df["obs_date"]).dt.date
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).copy()

    if df.empty:
        raise ValueError(f"FRED series {symbol!r} returned no usable data rows")

    return df.sort_values("obs_date").reset_index(drop=True)


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
        return _parse_fred_response(text, symbol)
