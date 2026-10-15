"""Ingestion engine — PriceSink interface and run_backfill logic.

Sources are tried in ascending priority order per instrument; the first
successful one is used for the whole request (no row splicing across providers).
bloomberg_ews sources are always skipped — they load via ``mad-data load-ews``.

Each instrument is wrapped in a savepoint when a SQLAlchemy session is
provided, so a failed write rolls back only that instrument's rows without
poisoning the outer transaction.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from datetime import date
from typing import Any, Optional, Protocol, runtime_checkable

import pandas as pd

from .db.known_at import compute_known_at
from .registry import Instrument


@runtime_checkable
class PriceSink(Protocol):
    """Write-side abstraction for the price store.

    Postgres now; a Kafka producer in Phase 8.
    """

    def write(self, rows: list[dict]) -> int:
        """Upsert *rows* and return the count of rows inserted or changed."""
        ...


class PostgresPriceSink:
    """PriceSink backed by the Postgres upsert_prices repository function."""

    def __init__(self, session: Any) -> None:
        self._session = session

    def write(self, rows: list[dict]) -> int:
        from .db.repository import upsert_prices

        return upsert_prices(self._session, rows)


@dataclass
class InstrumentResult:
    """Outcome of a single instrument fetch-and-store attempt."""

    instrument_id: str
    provider: Optional[str]  # which provider succeeded; None on error/skip
    status: str  # "ok" | "error" | "skipped"
    rows_written: int
    first_date: Optional[date]
    last_date: Optional[date]
    error: Optional[str]


def run_backfill(
    instruments: list[Instrument],
    *,
    providers: dict[str, Any],
    sink: PriceSink,
    session: Any = None,
    run_id: Optional[int] = None,
    since: Optional[date] = None,
) -> list[InstrumentResult]:
    """Fetch and store prices for *instruments* using *providers* and *sink*.

    Parameters
    ----------
    instruments:
        Active instruments to process.
    providers:
        Mapping of provider name to provider instance, e.g.
        ``{"fred": FredProvider(), "yahoo": YahooProvider(...)}``.
    sink:
        Where to write rows — ``PostgresPriceSink`` in production,
        a test double offline.
    session:
        SQLAlchemy ``Session``, or ``None`` for offline use.  When provided,
        each instrument is wrapped in a savepoint.
    run_id:
        If provided, stored on every price row as ``run_id``.
    since:
        Fetch data from this date onward.  ``None`` = full history.
    """
    results: list[InstrumentResult] = []
    for instrument in instruments:
        result = _process_one(instrument, providers, sink, run_id, since, session)
        results.append(result)
    return results


# ── Internal helpers ──────────────────────────────────────────────────────────


def _process_one(
    instrument: Instrument,
    providers: dict[str, Any],
    sink: PriceSink,
    run_id: Optional[int],
    since: Optional[date],
    session: Any,
) -> InstrumentResult:
    live_sources = sorted(
        [s for s in instrument.sources if s.provider != "bloomberg_ews" and s.symbol],
        key=lambda s: s.priority,
    )

    if not live_sources:
        return InstrumentResult(
            instrument_id=instrument.id,
            provider=None,
            status="skipped",
            rows_written=0,
            first_date=None,
            last_date=None,
            error="no live sources configured",
        )

    errors: list[str] = []
    last_provider: Optional[str] = None

    for source in live_sources:
        last_provider = source.provider
        provider = providers.get(source.provider)
        if provider is None:
            errors.append(f"{source.provider}: not in providers map")
            continue

        freq = (
            source.source_frequency.value
            if source.source_frequency is not None
            else instrument.native_frequency.value
        )

        rows: list[dict] = []
        n_written = 0
        error_msg: Optional[str] = None

        ctx = session.begin_nested() if session is not None else contextlib.nullcontext()
        try:
            with ctx:
                df = provider.get_history(source.symbol, start=since)
                rows = _to_price_rows(
                    df,
                    instrument_id=instrument.id,
                    provider=source.provider,
                    freq=freq,
                    lag=source.publication_lag_days,
                    run_id=run_id,
                )
                n_written = sink.write(rows) if rows else 0
        except Exception as exc:
            error_msg = str(exc)[:500]

        if error_msg is None:
            return InstrumentResult(
                instrument_id=instrument.id,
                provider=source.provider,
                status="ok",
                rows_written=n_written,
                first_date=rows[0]["obs_date"] if rows else None,
                last_date=rows[-1]["obs_date"] if rows else None,
                error=None,
            )
        errors.append(f"{source.provider}/{source.symbol}: {error_msg}")

    return InstrumentResult(
        instrument_id=instrument.id,
        provider=last_provider,
        status="error",
        rows_written=0,
        first_date=None,
        last_date=None,
        error="; ".join(errors)[:500] if errors else "all sources exhausted",
    )


def _to_price_rows(
    df: pd.DataFrame,
    *,
    instrument_id: str,
    provider: str,
    freq: str,
    lag: int,
    run_id: Optional[int],
) -> list[dict]:
    return [
        {
            "instrument_id": instrument_id,
            "provider": provider,
            "obs_date": row.obs_date,
            "value": row.value,
            "known_at": compute_known_at(row.obs_date, freq, lag),
            "run_id": run_id,
        }
        for row in df.itertuples(index=False)
    ]
