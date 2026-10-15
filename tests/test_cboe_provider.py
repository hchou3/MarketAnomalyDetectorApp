"""Tests for CboeProvider."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from market_anomaly_detector.providers.cboe import _CDN_URLS, CboeProvider

FIXTURES = Path(__file__).parent / "fixtures" / "cboe"


def _load(filename: str) -> str:
    return (FIXTURES / filename).read_text()


# ── CboeProvider unit tests (no network) ─────────────────────────────────────


def test_cboe_get_history_columns():
    provider = CboeProvider(_fetcher=lambda url: _load("VIX_History.csv"))
    hist = provider.get_history("VIX")
    assert list(hist.columns) == ["obs_date", "value"]


def test_cboe_obs_date_are_date_objects():
    provider = CboeProvider(_fetcher=lambda url: _load("VIX_History.csv"))
    hist = provider.get_history("VIX")
    assert isinstance(hist["obs_date"].iloc[0], date)


def test_cboe_sorted_ascending():
    provider = CboeProvider(_fetcher=lambda url: _load("VIX_History.csv"))
    hist = provider.get_history("VIX")
    assert hist["obs_date"].is_monotonic_increasing


def test_cboe_no_nans():
    provider = CboeProvider(_fetcher=lambda url: _load("VIX3M_History.csv"))
    hist = provider.get_history("VIX3M")
    assert hist["value"].notna().all()


def test_cboe_date_filter_start():
    # VIX fixture: 01-02, 01-03, 01-04, 01-05, 01-08, 01-09, 01-10
    provider = CboeProvider(_fetcher=lambda url: _load("VIX_History.csv"))
    hist = provider.get_history("VIX", start=date(2024, 1, 4))
    assert all(d >= date(2024, 1, 4) for d in hist["obs_date"])
    assert len(hist) == 5  # 01-04, 01-05, 01-08, 01-09, 01-10


def test_cboe_date_filter_end():
    provider = CboeProvider(_fetcher=lambda url: _load("VIX_History.csv"))
    hist = provider.get_history("VIX", end=date(2024, 1, 4))
    assert all(d <= date(2024, 1, 4) for d in hist["obs_date"])
    assert len(hist) == 3  # 3 rows on or before 2024-01-04


def test_cboe_date_filter_start_and_end():
    provider = CboeProvider(_fetcher=lambda url: _load("VIX_History.csv"))
    hist = provider.get_history("VIX", start=date(2024, 1, 3), end=date(2024, 1, 5))
    assert len(hist) == 3


def test_cboe_fallback_to_second_url():
    """First CDN failing must trigger a retry on the secondary CDN URL."""
    call_count = [0]
    content = _load("VIX3M_History.csv")

    def fetcher(url: str) -> str:
        call_count[0] += 1
        if "cdn-api" not in url and call_count[0] == 1:
            raise RuntimeError("primary CDN unavailable")
        return content

    provider = CboeProvider(_fetcher=fetcher)
    hist = provider.get_history("VIX3M")
    assert not hist.empty
    assert call_count[0] == 2
    assert hist["obs_date"].iloc[0] == date(2024, 1, 2)


def test_cboe_primary_url_is_tried_first():
    """Primary CDN URL must be the first one attempted."""
    first_url = [None]

    def fetcher(url: str) -> str:
        if first_url[0] is None:
            first_url[0] = url
        return _load("VIX_History.csv")

    CboeProvider(_fetcher=fetcher).get_history("VIX")
    assert _CDN_URLS[0].format(symbol="VIX") == first_url[0]


def test_cboe_both_urls_fail_raises():
    def fetcher(url: str) -> str:
        raise RuntimeError("CDN unavailable")

    provider = CboeProvider(_fetcher=fetcher)
    with pytest.raises(RuntimeError, match="All Cboe CDN URLs failed"):
        provider.get_history("VIX")


def test_cboe_empty_csv_raises():
    empty_csv = "Date,Open,High,Low,Close,Volume\n"
    provider = CboeProvider(_fetcher=lambda url: empty_csv)
    with pytest.raises(ValueError, match="no data"):
        provider.get_history("VIX")


# ── Network tests ─────────────────────────────────────────────────────────────


@pytest.mark.network
def test_cboe_live_vix():
    provider = CboeProvider()
    hist = provider.get_history("VIX", start=date(2024, 1, 1), end=date(2024, 3, 31))
    assert not hist.empty
    assert hist["value"].notna().all()


@pytest.mark.network
def test_cboe_live_vvix():
    provider = CboeProvider()
    hist = provider.get_history("VVIX", start=date(2024, 1, 1), end=date(2024, 3, 31))
    assert not hist.empty
