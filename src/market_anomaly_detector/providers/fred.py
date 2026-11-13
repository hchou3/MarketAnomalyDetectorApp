"""FRED provider — fetches time-series from the St. Louis Fed.

When ``FRED_API_KEY`` is set in the environment, the official JSON API at
``api.stlouisfed.org`` is used (authenticated, reliable from any network).
Without a key the public fredgraph.csv endpoint is used instead.

Missing values (".") are dropped in both paths.  ``FRED_API_KEY`` is never
logged, printed, or stored; it is redacted from any exception text that could
reach ingestion_run_items.error.
"""

from __future__ import annotations

import json
import os
import re
from datetime import date
from typing import Callable, Optional
from urllib.parse import urlencode

import pandas as pd

from ._http import http_get

_FREDGRAPH_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_API_KEY_RE = re.compile(r"api_key=[^&\s]+")


def _redact(text: str) -> str:
    """Replace api_key=<value> in any string to prevent key leakage."""
    return _API_KEY_RE.sub("api_key=REDACTED", text)


def _parse_fred_json(text: str, symbol: str) -> pd.DataFrame:
    """Parse a FRED JSON API response into a two-column DataFrame.

    ``value == "."`` observations (missing data) are dropped.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"FRED API for {symbol!r}: invalid JSON response") from exc

    rows: list[tuple[str, str]] = []
    for obs in data.get("observations", []):
        date_str = obs.get("date", "")
        value_str = obs.get("value", "")
        if _DATE_RE.match(date_str) and value_str != ".":
            rows.append((date_str, value_str))

    if not rows:
        raise ValueError(f"FRED series {symbol!r} returned no usable data rows")

    df = pd.DataFrame(rows, columns=["obs_date", "value"])
    df["obs_date"] = pd.to_datetime(df["obs_date"]).dt.date
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).copy()

    if df.empty:
        raise ValueError(f"FRED series {symbol!r} returned no usable data rows")

    return df.sort_values("obs_date").reset_index(drop=True)


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
    """Fetch data from FRED.

    When ``FRED_API_KEY`` is present in the environment the official JSON API
    is used; otherwise the public fredgraph.csv endpoint is used.  Missing
    values (".") are dropped in both paths.  An empty or all-missing series
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
        """Return DataFrame[obs_date: date, value: float] for one FRED series."""
        api_key = os.environ.get("FRED_API_KEY")
        if api_key:
            return self._get_history_api(symbol, start, api_key)
        return self._get_history_csv(symbol, start, end)

    def _get_history_api(
        self,
        symbol: str,
        start: Optional[date],
        api_key: str,
    ) -> pd.DataFrame:
        params: dict[str, str] = {
            "series_id": symbol,
            "api_key": api_key,
            "file_type": "json",
        }
        if start is not None:
            params["observation_start"] = start.isoformat()
        url = f"{_FRED_API_URL}?{urlencode(params)}"
        try:
            text = self._fetch(url)
        except Exception as exc:
            raise RuntimeError(f"FRED API error for {symbol!r}: {_redact(str(exc))}") from None
        return _parse_fred_json(text, symbol)

    def _get_history_csv(
        self,
        symbol: str,
        start: Optional[date],
        end: Optional[date],
    ) -> pd.DataFrame:
        params: dict[str, str] = {"id": symbol}
        if start is not None:
            params["cosd"] = start.isoformat()
        if end is not None:
            params["coed"] = end.isoformat()
        url = f"{_FREDGRAPH_URL}?{urlencode(params)}"
        text = self._fetch(url)
        return _parse_fred_response(text, symbol)
