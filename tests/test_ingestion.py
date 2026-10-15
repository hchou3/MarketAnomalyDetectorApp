"""Offline tests for the ingestion engine.

All tests run without a database — providers are replaced with FakeProvider
instances and writes go to in-memory sinks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

import pandas as pd

from market_anomaly_detector.ingestion import run_backfill
from market_anomaly_detector.registry import (
    AssetClass,
    Fidelity,
    Frequency,
    Instrument,
    SourceEntry,
    Units,
)

# ── Test doubles ──────────────────────────────────────────────────────────────


def _src(
    provider: str,
    symbol: str,
    priority: int,
    freq: Optional[str] = None,
    lag: int = 0,
    auto_adjust: bool = False,
) -> SourceEntry:
    return SourceEntry(
        provider=provider,
        symbol=symbol,
        priority=priority,
        fidelity=Fidelity.exact,
        source_frequency=Frequency(freq) if freq else None,
        publication_lag_days=lag,
        auto_adjust=auto_adjust,
    )


def _inst(
    id_: str,
    sources: list[SourceEntry],
    native_freq: str = "daily",
) -> Instrument:
    return Instrument(
        id=id_,
        name=f"Test {id_}",
        asset_class=AssetClass.volatility,
        groups=["test"],
        units=Units.index_points,
        quote="points",
        native_frequency=Frequency(native_freq),
        sources=sources,
    )


def _df(rows: list[tuple[str, float]]) -> pd.DataFrame:
    """Build a provider-style DataFrame from (iso_date, value) tuples."""
    return pd.DataFrame(
        {
            "obs_date": [date.fromisoformat(d) for d, _ in rows],
            "value": [float(v) for _, v in rows],
        }
    )


class FakeProvider:
    """Provider stub whose responses are pre-configured DataFrames or exceptions."""

    def __init__(self, name: str, responses: dict[str, Any]) -> None:
        self.name = name
        self._responses = responses  # {symbol: DataFrame | Exception}

    def get_history(
        self,
        symbol: str,
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> pd.DataFrame:
        resp = self._responses.get(symbol, ValueError(f"no fixture for {symbol!r}"))
        if isinstance(resp, BaseException):
            raise type(resp)(str(resp))
        df: pd.DataFrame = resp
        if start is not None:
            df = df[df["obs_date"] >= start]
        return df.reset_index(drop=True)


@dataclass
class CapturingSink:
    """Sink that accepts every row unconditionally."""

    rows: list[dict] = field(default_factory=list)
    write_calls: int = 0

    def write(self, rows: list[dict]) -> int:
        self.write_calls += 1
        self.rows.extend(rows)
        return len(rows)


@dataclass
class MemorySink:
    """Sink that simulates upsert semantics (deduplicates by primary key)."""

    _seen: set = field(default_factory=set)
    rows: list[dict] = field(default_factory=list)

    def write(self, rows: list[dict]) -> int:
        new = 0
        for row in rows:
            key = (row["instrument_id"], row["provider"], row["obs_date"])
            if key not in self._seen:
                self._seen.add(key)
                self.rows.append(row)
                new += 1
        return new


# ── Shared fixture data ───────────────────────────────────────────────────────

_THREE_ROWS = [("2024-01-02", 100.0), ("2024-01-03", 101.0), ("2024-01-04", 102.0)]


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_backfill_ok_result_fields():
    inst = _inst("gold", [_src("fred", "DGS10", 1)])
    sink = CapturingSink()

    results = run_backfill(
        [inst],
        providers={"fred": FakeProvider("fred", {"DGS10": _df(_THREE_ROWS)})},
        sink=sink,
    )

    assert len(results) == 1
    r = results[0]
    assert r.status == "ok"
    assert r.provider == "fred"
    assert r.instrument_id == "gold"
    assert r.rows_written == 3
    assert r.first_date == date(2024, 1, 2)
    assert r.last_date == date(2024, 1, 4)
    assert r.error is None


def test_backfill_idempotency():
    """Second run with a MemorySink (upsert semantics) must write 0 new rows."""
    inst = _inst("gold", [_src("fred", "DGS10", 1)])
    provider = FakeProvider("fred", {"DGS10": _df(_THREE_ROWS)})
    sink = MemorySink()

    r1 = run_backfill([inst], providers={"fred": provider}, sink=sink)
    r2 = run_backfill([inst], providers={"fred": provider}, sink=sink)

    assert r1[0].rows_written == 3
    assert r2[0].rows_written == 0
    assert len(sink.rows) == 3  # total unique rows unchanged


def test_backfill_failure_isolation():
    """A failing instrument must not prevent other instruments from being written."""
    inst_a = _inst("inst_a", [_src("fred", "BAD", 1)])
    inst_b = _inst("inst_b", [_src("fred", "GOOD", 1)])
    fred = FakeProvider(
        "fred",
        {
            "BAD": ValueError("provider unavailable"),
            "GOOD": _df(_THREE_ROWS),
        },
    )
    sink = CapturingSink()

    results = run_backfill([inst_a, inst_b], providers={"fred": fred}, sink=sink)

    r_a = next(r for r in results if r.instrument_id == "inst_a")
    r_b = next(r for r in results if r.instrument_id == "inst_b")

    assert r_a.status == "error"
    assert r_a.rows_written == 0
    assert r_b.status == "ok"
    assert r_b.rows_written == 3

    written_ids = {row["instrument_id"] for row in sink.rows}
    assert written_ids == {"inst_b"}


def test_backfill_fallback_to_priority_2():
    """When priority 1 fails, priority 2 is used and recorded in the result."""
    inst = _inst(
        "us_10y",
        [
            _src("fred", "DGS10", 1),
            _src("yahoo", "^TNX", 2),
        ],
    )
    fred = FakeProvider("fred", {"DGS10": ValueError("FRED down")})
    yahoo = FakeProvider("yahoo", {"^TNX": _df(_THREE_ROWS)})
    sink = CapturingSink()

    results = run_backfill([inst], providers={"fred": fred, "yahoo": yahoo}, sink=sink)

    r = results[0]
    assert r.status == "ok"
    assert r.provider == "yahoo"
    assert r.rows_written == 3
    assert all(row["provider"] == "yahoo" for row in sink.rows)


def test_backfill_all_sources_fail():
    """When every source fails, the result has status='error' and zero rows."""
    inst = _inst(
        "us_10y",
        [
            _src("fred", "DGS10", 1),
            _src("yahoo", "^TNX", 2),
        ],
    )
    sink = CapturingSink()

    results = run_backfill(
        [inst],
        providers={
            "fred": FakeProvider("fred", {"DGS10": ValueError("down")}),
            "yahoo": FakeProvider("yahoo", {"^TNX": ValueError("down")}),
        },
        sink=sink,
    )

    r = results[0]
    assert r.status == "error"
    assert r.rows_written == 0
    assert not sink.rows


def test_backfill_skipped_when_no_live_sources():
    """Instruments with only bloomberg_ews sources are skipped gracefully."""
    inst = _inst(
        "ews_only",
        [SourceEntry(provider="bloomberg_ews", symbol="VIX", priority=1, fidelity=Fidelity.exact)],
    )
    sink = CapturingSink()

    results = run_backfill([inst], providers={}, sink=sink)

    r = results[0]
    assert r.status == "skipped"
    assert r.rows_written == 0
    assert not sink.rows


def test_backfill_monthly_known_at():
    """Monthly source: known_at is computed from period end, not obs_date.

    obs_date=2023-01-01 (FRED convention = January 2023) with lag=40:
      period end  = 2023-01-31
      known_at    = 2023-01-31 + 40 days = 2023-03-12
    """
    inst = _inst(
        "de_10y",
        [_src("fred", "IRLTLT01DEM156N", 1, freq="monthly", lag=40)],
        native_freq="monthly",
    )
    monthly_df = _df([("2023-01-01", 2.5), ("2023-02-01", 2.6)])
    sink = CapturingSink()

    run_backfill(
        [inst],
        providers={"fred": FakeProvider("fred", {"IRLTLT01DEM156N": monthly_df})},
        sink=sink,
    )

    jan = next(r for r in sink.rows if r["obs_date"] == date(2023, 1, 1))
    assert jan["known_at"] == date(2023, 3, 12)

    # 2023-02-28 + 40 days = 2023-04-09
    feb = next(r for r in sink.rows if r["obs_date"] == date(2023, 2, 1))
    assert feb["known_at"] == date(2023, 4, 9)


def test_backfill_since_forwarded_to_provider():
    """The ``since`` date must be passed as ``start`` to provider.get_history."""
    calls: list[dict] = []

    class TrackingProvider:
        name = "fred"

        def get_history(self, symbol: str, start=None, end=None) -> pd.DataFrame:
            calls.append({"symbol": symbol, "start": start})
            return _df(_THREE_ROWS)

    inst = _inst("gold", [_src("fred", "DGS10", 1)])
    since = date(2024, 1, 3)

    run_backfill(
        [inst],
        providers={"fred": TrackingProvider()},
        sink=CapturingSink(),
        since=since,
    )

    assert len(calls) == 1
    assert calls[0]["start"] == since


def test_backfill_run_id_stored_in_rows():
    """Every price row must carry the run_id passed to run_backfill."""
    inst = _inst("gold", [_src("fred", "DGS10", 1)])
    sink = CapturingSink()

    run_backfill(
        [inst],
        providers={"fred": FakeProvider("fred", {"DGS10": _df(_THREE_ROWS)})},
        sink=sink,
        run_id=42,
    )

    assert all(row["run_id"] == 42 for row in sink.rows)


def test_backfill_daily_known_at_zero_lag():
    """Daily source with lag=0: known_at equals obs_date."""
    inst = _inst("gold", [_src("yahoo", "GC=F", 1, lag=0)])
    sink = CapturingSink()

    run_backfill(
        [inst],
        providers={"yahoo": FakeProvider("yahoo", {"GC=F": _df(_THREE_ROWS)})},
        sink=sink,
    )

    for row in sink.rows:
        assert row["known_at"] == row["obs_date"]


def test_backfill_error_message_contains_provider():
    """The error string on a failed result must name the provider that was tried."""
    inst = _inst("gold", [_src("fred", "DGS10", 1)])
    sink = CapturingSink()

    results = run_backfill(
        [inst],
        providers={"fred": FakeProvider("fred", {"DGS10": ValueError("timeout")})},
        sink=sink,
    )

    r = results[0]
    assert r.status == "error"
    assert "fred" in (r.error or "")


def test_backfill_missing_provider_in_map():
    """If a provider is not in the providers map, the instrument is an error."""
    inst = _inst("gold", [_src("cboe", "VIX", 1)])
    sink = CapturingSink()

    results = run_backfill([inst], providers={}, sink=sink)

    r = results[0]
    assert r.status == "error"
    assert r.rows_written == 0
