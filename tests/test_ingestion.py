"""Tests for the ingestion engine.

Most tests are offline (no DB) — providers are replaced with FakeProvider
instances and writes go to in-memory sinks.  The ``@pytest.mark.db`` tests
at the bottom exercise PostgresPriceSink against a real database.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

import pandas as pd
import pytest

from market_anomaly_detector.ingestion import is_stale, run_backfill
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
    asset_class: AssetClass = AssetClass.volatility,
) -> Instrument:
    return Instrument(
        id=id_,
        name=f"Test {id_}",
        asset_class=asset_class,
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

    def write(self, rows: list[dict]) -> tuple[int, int, None, None]:
        self.write_calls += 1
        self.rows.extend(rows)
        return (len(rows), 0, None, None)


@dataclass
class MemorySink:
    """Sink that simulates upsert semantics (deduplicates by primary key)."""

    _seen: set = field(default_factory=set)
    rows: list[dict] = field(default_factory=list)

    def write(self, rows: list[dict]) -> tuple[int, int, None, None]:
        new = 0
        for row in rows:
            key = (row["instrument_id"], row["provider"], row["obs_date"])
            if key not in self._seen:
                self._seen.add(key)
                self.rows.append(row)
                new += 1
        return (new, 0, None, None)


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
    assert r.rows_written == 3  # property: rows_new + rows_revised
    assert r.rows_new == 3
    assert r.rows_revised == 0
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


def test_backfill_get_start_overrides_since():
    """When get_start is provided, it takes precedence over since."""
    calls: list[dict] = []

    class TrackingProvider:
        name = "fred"

        def get_history(self, symbol: str, start=None, end=None) -> pd.DataFrame:
            calls.append({"symbol": symbol, "start": start})
            return _df(_THREE_ROWS)

    inst = _inst("gold", [_src("fred", "DGS10", 1)])
    override_date = date(2024, 6, 1)

    run_backfill(
        [inst],
        providers={"fred": TrackingProvider()},
        sink=CapturingSink(),
        since=date(2024, 1, 1),
        get_start=lambda inst_id, provider: override_date,
    )

    assert calls[0]["start"] == override_date


def test_backfill_get_start_none_means_full_history():
    """get_start returning None triggers a full history fetch (start=None)."""
    calls: list[dict] = []

    class TrackingProvider:
        name = "fred"

        def get_history(self, symbol: str, start=None, end=None) -> pd.DataFrame:
            calls.append({"symbol": symbol, "start": start})
            return _df(_THREE_ROWS)

    inst = _inst("gold", [_src("fred", "DGS10", 1)])

    run_backfill(
        [inst],
        providers={"fred": TrackingProvider()},
        sink=CapturingSink(),
        get_start=lambda inst_id, provider: None,
    )

    assert calls[0]["start"] is None


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


# ── is_stale ──────────────────────────────────────────────────────────────────


def test_staleness_daily_not_stale():
    """Daily: 4 business days = not stale."""
    today = date(2024, 1, 10)  # Wednesday
    last = date(2024, 1, 4)  # Thu 4, Fri 5, Mon 8, Tue 9 = 4 bdays to Jan 10
    assert not is_stale(last, "daily", today)


def test_staleness_daily_stale_at_boundary():
    """Daily: 5 business days = stale."""
    today = date(2024, 1, 10)  # Wednesday
    last = date(2024, 1, 3)  # Wed → Thu 4, Fri 5, Mon 8, Tue 9, Wed 10 = 5 bdays
    assert is_stale(last, "daily", today)


def test_staleness_weekly_not_stale():
    """Weekly: 9 calendar days = not stale."""
    today = date(2024, 1, 20)
    last = date(2024, 1, 11)  # 9 days
    assert not is_stale(last, "weekly", today)


def test_staleness_weekly_stale_at_boundary():
    """Weekly: 10 calendar days = stale."""
    today = date(2024, 1, 20)
    last = date(2024, 1, 10)  # 10 days
    assert is_stale(last, "weekly", today)


def test_staleness_monthly_not_stale():
    """Monthly: 44 calendar days = not stale."""
    today = date(2024, 3, 15)
    last = date(2024, 1, 31)  # Feb(29) + 15 Mar = 44 days
    assert not is_stale(last, "monthly", today)


def test_staleness_monthly_stale_at_boundary():
    """Monthly: 45 calendar days = stale."""
    today = date(2024, 3, 15)
    last = date(2024, 1, 30)  # 1 Jan + 29 Feb + 15 Mar = 45 days
    assert is_stale(last, "monthly", today)


def test_staleness_no_data_always_stale():
    assert is_stale(None, "daily", date(2024, 1, 10))


def test_staleness_publication_lag_shifts_threshold():
    """A daily series that would be stale without lag is not stale when lag covers the gap."""
    today = date(2024, 1, 10)  # Wednesday
    # 5 bdays behind today → stale without lag
    last = date(2024, 1, 3)
    assert is_stale(last, "daily", today, publication_lag_days=0)
    # With 3-day publication lag, effective_today = Jan 7 → only 2 bdays from last
    assert not is_stale(last, "daily", today, publication_lag_days=3)


def test_staleness_monthly_with_lag_not_stale():
    """Monthly series with 45-day lag: 44 days ago is not stale despite threshold being 45."""
    today = date(2024, 3, 15)
    last = date(2024, 1, 31)  # 44 days before today → not stale at 0 lag
    # With 10-day lag: effective_today = Mar 5; (Mar 5 - Jan 31) = 34 days < 45 → not stale
    assert not is_stale(last, "monthly", today, publication_lag_days=10)


# ── DB tests (require Postgres) ───────────────────────────────────────────────


def _insert_instrument_row(session, instrument_id: str):
    """Insert a minimal InstrumentRow directly (bypasses registry sync)."""
    from market_anomaly_detector.db.models import InstrumentRow

    row = InstrumentRow(
        id=instrument_id,
        name=f"Test {instrument_id}",
        asset_class="volatility",
        units="index_points",
        quote="points",
        native_frequency="daily",
        notes="",
        active=True,
    )
    session.add(row)
    session.flush()
    return row


@pytest.mark.db
def test_backfill_db_savepoint_isolates_write_failure(db_session):
    """DB-level write failure rolls back only that instrument's rows.

    Scenario: two instruments, same Yahoo provider.  The first returns a value
    that overflows Numeric(20, 8) — ``1e15`` has 16 integer digits, exceeding
    the 12-integer-digit limit.  The second returns valid data.

    Expected outcomes:
    - ``inst_overflow`` result: status="error", rows_written=0, 0 rows in DB
    - ``inst_normal`` result:   status="ok",    rows_written=2, 2 rows in DB
    - Run items record both outcomes correctly
    """
    from sqlalchemy import func, select

    from market_anomaly_detector.db.models import IngestionRunItemRow, PriceRow
    from market_anomaly_detector.db.repository import add_run_item, start_run
    from market_anomaly_detector.ingestion import PostgresPriceSink, run_backfill

    _insert_instrument_row(db_session, "inst_overflow")
    _insert_instrument_row(db_session, "inst_normal")
    db_session.flush()

    inst_overflow = _inst("inst_overflow", [_src("yahoo", "OVER=X", 1)])
    inst_normal = _inst("inst_normal", [_src("yahoo", "NORM=X", 1)])

    # 1e15 = 10^15 overflows Numeric(20, 8) (max 12 integer digits)
    overflow_df = pd.DataFrame({"obs_date": [date(2024, 1, 2)], "value": [1e15]})
    normal_df = pd.DataFrame(
        {
            "obs_date": [date(2024, 1, 2), date(2024, 1, 3)],
            "value": [100.0, 101.0],
        }
    )
    providers = {
        "yahoo": FakeProvider(
            "yahoo",
            {"OVER=X": overflow_df, "NORM=X": normal_df},
        )
    }

    run = start_run(db_session, mode="backfill")
    db_session.flush()

    sink = PostgresPriceSink(db_session)
    results = run_backfill(
        [inst_overflow, inst_normal],
        providers=providers,
        sink=sink,
        session=db_session,
        run_id=run.id,
    )

    # Record run items from results (mirrors what the CLI does)
    for r in results:
        if r.status == "ok":
            add_run_item(
                db_session,
                run,
                r.instrument_id,
                r.provider,
                "ok",
                rows_new=r.rows_new,
                rows_revised=r.rows_revised,
                first_date=r.first_date,
                last_date=r.last_date,
            )
        else:
            add_run_item(
                db_session,
                run,
                r.instrument_id,
                r.provider or "yahoo",
                "error",
                error=r.error,
            )
    db_session.flush()

    # ── InstrumentResult assertions ───────────────────────────────────────────
    r_over = next(r for r in results if r.instrument_id == "inst_overflow")
    r_norm = next(r for r in results if r.instrument_id == "inst_normal")

    assert r_over.status == "error"
    assert r_over.rows_written == 0
    assert r_norm.status == "ok"
    assert r_norm.rows_written == 2
    assert r_norm.rows_new == 2
    assert r_norm.rows_revised == 0
    assert r_norm.max_abs_revision is None
    assert r_norm.max_rel_revision is None

    # ── DB state: overflow instrument has no rows ─────────────────────────────
    overflow_count = db_session.execute(
        select(func.count()).select_from(PriceRow).where(PriceRow.instrument_id == "inst_overflow")
    ).scalar_one()
    assert overflow_count == 0, "Overflow rows must be rolled back by the savepoint"

    normal_count = db_session.execute(
        select(func.count()).select_from(PriceRow).where(PriceRow.instrument_id == "inst_normal")
    ).scalar_one()
    assert normal_count == 2

    # ── Run items: one error entry, one ok entry ──────────────────────────────
    error_item = db_session.execute(
        select(IngestionRunItemRow).where(
            IngestionRunItemRow.run_id == run.id,
            IngestionRunItemRow.instrument_id == "inst_overflow",
        )
    ).scalar_one()
    assert error_item.status == "error"
    assert error_item.rows_written == 0

    ok_item = db_session.execute(
        select(IngestionRunItemRow).where(
            IngestionRunItemRow.run_id == run.id,
            IngestionRunItemRow.instrument_id == "inst_normal",
        )
    ).scalar_one()
    assert ok_item.status == "ok"
    assert ok_item.rows_written == 2
    assert ok_item.rows_new == 2
    assert ok_item.rows_revised == 0


@pytest.mark.db
def test_backfill_repeated_unchanged_data_zero_revised(db_session):
    """Re-running backfill with identical provider data must produce 0 revised rows.

    Verifies the no-look-ahead guarantee for non-adjusted indices: a second
    identical fetch must not trigger spurious revisions (rows_revised == 0 and
    max_abs_revision is None).
    """
    from market_anomaly_detector.ingestion import PostgresPriceSink

    _insert_instrument_row(db_session, "gspc_test")
    db_session.flush()

    inst = _inst(
        "gspc_test",
        [_src("yahoo", "^GSPC", 1, auto_adjust=False)],
        asset_class=AssetClass.equity_index,
    )
    fixture_data = [
        ("2024-01-02", 4700.0),
        ("2024-01-03", 4710.5),
        ("2024-01-04", 4695.2),
        ("2024-01-05", 4720.0),
        ("2024-01-08", 4730.1),
    ]
    provider = FakeProvider("yahoo", {"^GSPC": _df(fixture_data)})
    sink = PostgresPriceSink(db_session)

    r1 = run_backfill([inst], providers={"yahoo": provider}, sink=sink, session=db_session)
    db_session.flush()
    assert r1[0].status == "ok"
    assert r1[0].rows_new == 5
    assert r1[0].rows_revised == 0
    assert r1[0].max_abs_revision is None

    r2 = run_backfill([inst], providers={"yahoo": provider}, sink=sink, session=db_session)
    db_session.flush()
    assert r2[0].status == "ok"
    assert r2[0].rows_new == 0
    assert r2[0].rows_revised == 0
    assert r2[0].max_abs_revision is None
    assert r2[0].max_rel_revision is None
