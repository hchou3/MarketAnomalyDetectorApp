"""Tests for FredProvider and the shared http_get retry helper."""

from __future__ import annotations

import urllib.error
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from market_anomaly_detector.providers.fred import FredProvider

FIXTURES = Path(__file__).parent / "fixtures" / "fred"


def _load(filename: str) -> str:
    return (FIXTURES / filename).read_text()


# ── FredProvider unit tests (no network) ─────────────────────────────────────


def test_fred_get_history_columns():
    provider = FredProvider(_fetcher=lambda url: _load("DGS10.csv"))
    hist = provider.get_history("DGS10")
    assert list(hist.columns) == ["obs_date", "value"]


def test_fred_obs_date_are_date_objects():
    provider = FredProvider(_fetcher=lambda url: _load("DGS10.csv"))
    hist = provider.get_history("DGS10")
    assert isinstance(hist["obs_date"].iloc[0], date)


def test_fred_drops_dot_missing_values():
    provider = FredProvider(_fetcher=lambda url: _load("DGS10.csv"))
    hist = provider.get_history("DGS10")
    # DGS10.csv has one "." row on 2024-01-05; it must be dropped
    assert hist["value"].notna().all()
    assert date(2024, 1, 5) not in hist["obs_date"].values


def test_fred_row_count_after_drop():
    provider = FredProvider(_fetcher=lambda url: _load("DGS10.csv"))
    hist = provider.get_history("DGS10")
    assert len(hist) == 6  # 7 rows minus 1 missing


def test_fred_sorted_ascending():
    provider = FredProvider(_fetcher=lambda url: _load("DGS10.csv"))
    hist = provider.get_history("DGS10")
    assert hist["obs_date"].is_monotonic_increasing


def test_fred_empty_series_raises():
    provider = FredProvider(_fetcher=lambda url: _load("EMPTY_SERIES.csv"))
    with pytest.raises(ValueError, match="no usable data"):
        provider.get_history("EMPTY_SERIES")


def test_fred_malformed_response_raises():
    # A single-column response (no comma) triggers the shape check
    provider = FredProvider(_fetcher=lambda url: "single_column_only")
    with pytest.raises(ValueError, match="Unexpected FRED response"):
        provider.get_history("BAD")


# ── Preamble stripping (new FRED format) ─────────────────────────────────────


def test_fred_preamble_stripped_columns():
    """New FRED format with Title/Series ID/Source/Release preamble parses correctly."""
    provider = FredProvider(_fetcher=lambda url: _load("DGS2_preamble.csv"))
    hist = provider.get_history("DGS2")
    assert list(hist.columns) == ["obs_date", "value"]


def test_fred_preamble_stripped_row_count():
    """Preamble rows are not counted as data rows."""
    provider = FredProvider(_fetcher=lambda url: _load("DGS2_preamble.csv"))
    hist = provider.get_history("DGS2")
    assert len(hist) == 4  # 5 rows minus 1 missing


def test_fred_preamble_stripped_drops_dot():
    """Missing-value rows are still dropped after preamble stripping."""
    provider = FredProvider(_fetcher=lambda url: _load("DGS2_preamble.csv"))
    hist = provider.get_history("DGS2")
    assert hist["value"].notna().all()
    assert date(2024, 1, 5) not in hist["obs_date"].values


def test_strip_preamble_passthrough():
    """Old-format CSV (no preamble) passes through unchanged."""
    from market_anomaly_detector.providers.fred import _strip_preamble

    old_fmt = "DATE,DGS10\n2024-01-02,3.95\n2024-01-03,3.97\n"
    assert _strip_preamble(old_fmt) == old_fmt


def test_strip_preamble_finds_date_line():
    """_strip_preamble returns text starting from the DATE header line."""
    from market_anomaly_detector.providers.fred import _strip_preamble

    new_fmt = "Title,Foo\nSeries ID,BAR\nDATE,BAR\n2024-01-01,1.0\n"
    result = _strip_preamble(new_fmt)
    assert result.startswith("DATE,BAR\n")
    assert "Title" not in result


# ── Date-range URL parameters ─────────────────────────────────────────────────


class _CapturingFetcher:
    """Captures the URL passed to it; returns the given content."""

    def __init__(self, content: str) -> None:
        self.calls: list[str] = []
        self._content = content

    def __call__(self, url: str) -> str:
        self.calls.append(url)
        return self._content


def test_fred_date_range_adds_cosd_coed():
    fetcher = _CapturingFetcher(_load("DEXJPUS.csv"))
    provider = FredProvider(_fetcher=fetcher)
    provider.get_history("DEXJPUS", start=date(2024, 1, 1), end=date(2024, 6, 30))
    assert len(fetcher.calls) == 1
    assert "cosd=2024-01-01" in fetcher.calls[0]
    assert "coed=2024-06-30" in fetcher.calls[0]


def test_fred_no_dates_omits_cosd_coed():
    fetcher = _CapturingFetcher(_load("DGS10.csv"))
    provider = FredProvider(_fetcher=fetcher)
    provider.get_history("DGS10")
    assert "cosd" not in fetcher.calls[0]
    assert "coed" not in fetcher.calls[0]


def test_fred_start_only_adds_cosd_not_coed():
    fetcher = _CapturingFetcher(_load("DGS10.csv"))
    provider = FredProvider(_fetcher=fetcher)
    provider.get_history("DGS10", start=date(2024, 1, 1))
    assert "cosd=2024-01-01" in fetcher.calls[0]
    assert "coed" not in fetcher.calls[0]


# ── http_get retry logic ──────────────────────────────────────────────────────


def test_http_get_retries_on_503(monkeypatch):
    from market_anomaly_detector.providers._http import http_get

    monkeypatch.setattr("time.sleep", lambda _: None)
    call_count = [0]

    def mock_urlopen(req, timeout, context):
        call_count[0] += 1
        if call_count[0] < 3:
            raise urllib.error.HTTPError(req.full_url, 503, "Service Unavailable", {}, None)
        resp = MagicMock()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        resp.read.return_value = b"DATE,DGS10\n2024-01-02,3.95"
        return resp

    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)
    result = http_get("https://example.com/data.csv")
    assert "DATE,DGS10" in result
    assert call_count[0] == 3


def test_http_get_raises_immediately_on_404(monkeypatch):
    from market_anomaly_detector.providers._http import http_get

    monkeypatch.setattr("time.sleep", lambda _: None)
    call_count = [0]

    def mock_urlopen(req, timeout, context):
        call_count[0] += 1
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        http_get("https://example.com/missing.csv")
    assert exc_info.value.code == 404
    assert call_count[0] == 1  # no retries on 4xx


def test_http_get_retries_on_429_with_retry_after(monkeypatch):
    from market_anomaly_detector.providers._http import http_get

    monkeypatch.setattr("time.sleep", lambda _: None)
    call_count = [0]

    def mock_urlopen(req, timeout, context):
        call_count[0] += 1
        if call_count[0] < 2:
            headers = MagicMock()
            headers.get.return_value = "0"
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", headers, None)
        resp = MagicMock()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        resp.read.return_value = b"DATE,DGS10\n2024-01-02,3.95"
        return resp

    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)
    result = http_get("https://example.com/data.csv")
    assert "DATE,DGS10" in result
    assert call_count[0] == 2


def test_http_get_retries_on_urlerror(monkeypatch):
    from market_anomaly_detector.providers._http import http_get

    monkeypatch.setattr("time.sleep", lambda _: None)
    call_count = [0]

    def mock_urlopen(req, timeout, context):
        call_count[0] += 1
        if call_count[0] < 2:
            raise urllib.error.URLError("Connection refused")
        resp = MagicMock()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        resp.read.return_value = b"DATE,DGS10\n2024-01-02,3.95"
        return resp

    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)
    result = http_get("https://example.com/data.csv")
    assert "DATE,DGS10" in result
    assert call_count[0] == 2


# ── Network tests (real FRED; skipped in offline CI) ─────────────────────────


@pytest.mark.network
def test_fred_live_dgs10():
    provider = FredProvider()
    hist = provider.get_history("DGS10", start=date(2024, 1, 1), end=date(2024, 3, 31))
    assert not hist.empty
    assert hist["obs_date"].iloc[0] == date(2024, 1, 2)
    assert hist["value"].notna().all()


@pytest.mark.network
def test_fred_live_dexjpus_range():
    provider = FredProvider()
    hist = provider.get_history("DEXJPUS", start=date(2024, 1, 1), end=date(2024, 1, 31))
    assert not hist.empty
    assert all(date(2024, 1, 1) <= d <= date(2024, 1, 31) for d in hist["obs_date"])
