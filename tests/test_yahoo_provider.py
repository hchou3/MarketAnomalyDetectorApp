"""Tests for YahooProvider."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from market_anomaly_detector.providers.yahoo import YahooProvider

FIXTURES = Path(__file__).parent / "fixtures" / "yahoo"


def _load_fixture(symbol: str) -> pd.DataFrame:
    path = FIXTURES / f"{symbol}.csv"
    return pd.read_csv(path, index_col=0, parse_dates=True)


def _make_fetcher(symbol: str):
    """Return a _fetcher that loads *symbol*'s fixture regardless of what get_history passes."""
    df = _load_fixture(symbol)

    def fetch(sym, start, end, auto_adjust):
        return df.copy()

    return fetch


def _make_empty_fetcher():
    def fetch(sym, start, end, auto_adjust):
        return pd.DataFrame()

    return fetch


# ── YahooProvider unit tests (no network) ────────────────────────────────────


def test_yahoo_get_history_columns():
    provider = YahooProvider(_fetcher=_make_fetcher("GC=F"))
    hist = provider.get_history("GC=F")
    assert list(hist.columns) == ["obs_date", "value"]


def test_yahoo_obs_date_are_date_objects():
    provider = YahooProvider(_fetcher=_make_fetcher("GC=F"))
    hist = provider.get_history("GC=F")
    assert isinstance(hist["obs_date"].iloc[0], date)


def test_yahoo_no_nans():
    provider = YahooProvider(_fetcher=_make_fetcher("^VIX"))
    hist = provider.get_history("^VIX")
    assert hist["value"].notna().all()


def test_yahoo_sorted_ascending():
    provider = YahooProvider(_fetcher=_make_fetcher("GC=F"))
    hist = provider.get_history("GC=F")
    assert hist["obs_date"].is_monotonic_increasing


def test_yahoo_exchange_local_dates_no_time_component():
    """obs_date must be a calendar date (no hour/minute), independent of timezone."""
    provider = YahooProvider(_fetcher=_make_fetcher("GC=F"))
    hist = provider.get_history("GC=F")
    # All dates must be pure date objects (no datetime, no tz)
    for d in hist["obs_date"]:
        assert isinstance(d, date)
        assert not isinstance(d, type(pd.Timestamp.now()))


def test_yahoo_tz_aware_index_is_stripped():
    """A tz-aware index in the fixture must be stripped to tz-naive exchange dates."""
    import pandas as pd

    df = _load_fixture("GC=F").copy()
    # Simulate yfinance returning UTC-aware index
    df.index = pd.to_datetime(df.index).tz_localize("UTC")

    def fetcher(sym, start, end, auto_adjust):
        return df

    provider = YahooProvider(_fetcher=fetcher)
    hist = provider.get_history("GC=F")
    assert hist["obs_date"].iloc[0] == date(2024, 1, 2)


def test_yahoo_empty_result_raises():
    provider = YahooProvider(_fetcher=_make_empty_fetcher())
    with pytest.raises(ValueError, match="no data"):
        provider.get_history("GC=F")


def test_yahoo_auto_adjust_passed_to_fetcher():
    """The auto_adjust value from symbol_auto_adjust must reach the _fetcher."""
    captured = [None]

    def fetcher(sym, start, end, auto_adjust):
        captured[0] = auto_adjust
        return _load_fixture("EWJ")

    provider = YahooProvider(
        symbol_auto_adjust={"EWJ": True},
        _fetcher=fetcher,
    )
    provider.get_history("EWJ")
    assert captured[0] is True


def test_yahoo_auto_adjust_defaults_to_false():
    captured = [None]

    def fetcher(sym, start, end, auto_adjust):
        captured[0] = auto_adjust
        return _load_fixture("GC=F")

    provider = YahooProvider(_fetcher=fetcher)
    provider.get_history("GC=F")
    assert captured[0] is False


def test_yahoo_date_filter_start():
    # GC=F fixture has rows on 01-02, 01-03, 01-04, 01-05, 01-08
    provider = YahooProvider(_fetcher=_make_fetcher("GC=F"))
    hist = provider.get_history("GC=F", start=date(2024, 1, 4))
    assert all(d >= date(2024, 1, 4) for d in hist["obs_date"])
    assert len(hist) == 3  # 01-04, 01-05, 01-08


def test_yahoo_date_filter_end():
    provider = YahooProvider(_fetcher=_make_fetcher("GC=F"))
    hist = provider.get_history("GC=F", end=date(2024, 1, 3))
    assert all(d <= date(2024, 1, 3) for d in hist["obs_date"])
    assert len(hist) == 2


def test_yahoo_etf_fixture_no_nans():
    """ETF fixture (auto_adjust=True format, fewer columns) must parse cleanly."""
    provider = YahooProvider(_fetcher=_make_fetcher("EWJ"))
    hist = provider.get_history("EWJ")
    assert hist["value"].notna().all()
    assert len(hist) == 5


def test_yahoo_request_delay_skipped_with_fetcher():
    """No time.sleep should be called when a test _fetcher is injected."""
    provider = YahooProvider(_fetcher=_make_fetcher("GC=F"), request_delay=5.0)
    # _request_delay is forced to 0.0 when _fetcher is provided
    assert provider._request_delay == 0.0


# ── Network tests ─────────────────────────────────────────────────────────────


@pytest.mark.network
def test_yahoo_live_gold_futures():
    provider = YahooProvider()
    hist = provider.get_history("GC=F", start=date(2024, 1, 1), end=date(2024, 3, 31))
    assert not hist.empty
    assert hist["value"].notna().all()
    assert hist["obs_date"].is_monotonic_increasing


@pytest.mark.network
def test_yahoo_live_etf_auto_adjust():
    """EWJ with auto_adjust=True should return values adjusted for dividends/splits."""
    provider = YahooProvider(symbol_auto_adjust={"EWJ": True})
    hist = provider.get_history("EWJ", start=date(2024, 1, 1), end=date(2024, 3, 31))
    assert not hist.empty
    assert hist["value"].notna().all()


@pytest.mark.network
def test_yahoo_live_vix_index():
    provider = YahooProvider()
    hist = provider.get_history("^VIX", start=date(2024, 1, 1), end=date(2024, 3, 31))
    assert not hist.empty
