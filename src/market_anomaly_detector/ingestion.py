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
from datetime import date, timedelta
from typing import Any, Callable, Optional, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from .db.known_at import compute_known_at
from .registry import Instrument


@runtime_checkable
class PriceSink(Protocol):
    """Write-side abstraction for the price store.

    Postgres now; a Kafka producer in Phase 8.
    """

    def write(self, rows: list[dict]) -> tuple[int, int, Optional[float], Optional[float]]:
        """Upsert *rows* and return ``(rows_new, rows_revised, max_abs, max_rel)``."""
        ...


class PostgresPriceSink:
    """PriceSink backed by the Postgres upsert_prices repository function."""

    def __init__(self, session: Any) -> None:
        self._session = session

    def write(self, rows: list[dict]) -> tuple[int, int, Optional[float], Optional[float]]:
        from .db.repository import upsert_prices

        r = upsert_prices(self._session, rows)
        return (r.rows_new, r.rows_revised, r.max_abs_revision, r.max_rel_revision)


@dataclass
class InstrumentResult:
    """Outcome of a single instrument fetch-and-store attempt."""

    instrument_id: str
    provider: Optional[str]  # which provider succeeded; None on error/skip
    status: str  # "ok" | "error" | "skipped"
    rows_new: int = 0
    rows_revised: int = 0
    first_date: Optional[date] = None
    last_date: Optional[date] = None
    error: Optional[str] = None
    max_abs_revision: Optional[float] = None
    max_rel_revision: Optional[float] = None

    @property
    def rows_written(self) -> int:
        return self.rows_new + self.rows_revised


def run_backfill(
    instruments: list[Instrument],
    *,
    providers: dict[str, Any],
    sink: PriceSink,
    session: Any = None,
    run_id: Optional[int] = None,
    since: Optional[date] = None,
    get_start: Optional[Callable[[str, str], Optional[date]]] = None,
    cutoff_date: Optional[date] = None,
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
        Ignored when *get_start* is provided.
    get_start:
        Optional callable ``(instrument_id, provider) -> date | None`` that
        returns the fetch start date for each (instrument, provider) pair.
        When it returns ``None``, the full history is fetched.
        Takes precedence over *since*.
    cutoff_date:
        Exclude daily bars with obs_date >= this date (default: today).
        Prevents incomplete intraday bars from being stored before the session
        closes.  Monthly/weekly bars are not filtered.
    """
    today_cutoff = cutoff_date if cutoff_date is not None else date.today()
    results: list[InstrumentResult] = []
    for instrument in instruments:
        result = _process_one(
            instrument, providers, sink, run_id, get_start, since, session, today_cutoff
        )
        results.append(result)
    return results


def is_stale(
    last_obs_date: Optional[date],
    native_frequency: str,
    today: Optional[date] = None,
    publication_lag_days: int = 0,
) -> bool:
    """Return True if the instrument is overdue for an update.

    Thresholds: daily → 5 business days; weekly → 10 calendar days;
    monthly → 75 calendar days.  ``publication_lag_days`` shifts the
    reference point backward so a series with a known release delay is
    not false-positively flagged.
    """
    if today is None:
        today = date.today()
    if last_obs_date is None:
        return True
    effective_today = today - timedelta(days=publication_lag_days)
    if effective_today <= last_obs_date:
        return False
    if native_frequency == "daily":
        n_bdays = int(np.busday_count(last_obs_date.isoformat(), effective_today.isoformat()))
        return n_bdays >= 5
    elif native_frequency == "weekly":
        return (effective_today - last_obs_date).days >= 10
    else:  # monthly
        return (effective_today - last_obs_date).days >= 75


# ── Internal helpers ──────────────────────────────────────────────────────────


def _process_one(
    instrument: Instrument,
    providers: dict[str, Any],
    sink: PriceSink,
    run_id: Optional[int],
    get_start: Optional[Callable[[str, str], Optional[date]]],
    since: Optional[date],
    session: Any,
    cutoff_date: Optional[date] = None,
) -> InstrumentResult:
    from .validation import validate_provider_output

    live_sources = sorted(
        [s for s in instrument.sources if s.provider != "bloomberg_ews" and s.symbol],
        key=lambda s: s.priority,
    )

    if not live_sources:
        return InstrumentResult(
            instrument_id=instrument.id,
            provider=None,
            status="skipped",
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

        effective_start = (
            get_start(instrument.id, source.provider) if get_start is not None else since
        )

        rows: list[dict] = []
        n_new = n_revised = 0
        error_msg: Optional[str] = None

        ctx = session.begin_nested() if session is not None else contextlib.nullcontext()
        try:
            with ctx:
                df = provider.get_history(source.symbol, start=effective_start)
                clean_df, flags = validate_provider_output(df, instrument)
                if cutoff_date is not None and freq == "daily":
                    clean_df = clean_df[clean_df["obs_date"] < cutoff_date].reset_index(drop=True)
                rows = _to_price_rows(
                    clean_df,
                    instrument_id=instrument.id,
                    provider=source.provider,
                    freq=freq,
                    lag=source.publication_lag_days,
                    run_id=run_id,
                )
                n_new, n_revised, max_abs, max_rel = (
                    sink.write(rows) if rows else (0, 0, None, None)
                )
                if flags and session is not None:
                    from .db.repository import write_quality_flags

                    write_quality_flags(
                        session,
                        flags,
                        instrument_id=instrument.id,
                        provider=source.provider,
                        run_id=run_id,
                    )
        except Exception as exc:
            error_msg = str(exc)[:500]

        if error_msg is None:
            return InstrumentResult(
                instrument_id=instrument.id,
                provider=source.provider,
                status="ok",
                rows_new=n_new,
                rows_revised=n_revised,
                first_date=rows[0]["obs_date"] if rows else None,
                last_date=rows[-1]["obs_date"] if rows else None,
                error=None,
                max_abs_revision=max_abs,
                max_rel_revision=max_rel,
            )
        errors.append(f"{source.provider}/{source.symbol}: {error_msg}")

    return InstrumentResult(
        instrument_id=instrument.id,
        provider=last_provider,
        status="error",
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
