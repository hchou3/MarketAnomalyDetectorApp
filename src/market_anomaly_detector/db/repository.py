"""Repository layer: thin wrappers over SQLAlchemy that own all DB writes.

All mutations go through these functions so the rest of the codebase never
constructs raw INSERT/UPDATE statements.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional

from sqlalchemy import delete, or_, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from ..registry import Instrument
from .models import (
    IngestionRunItemRow,
    IngestionRunRow,
    InstrumentGroupRow,
    InstrumentRow,
    LabelRow,
    PriceRow,
    SymbolMapRow,
)

# Postgres hard limit is 65,535 bind parameters per statement.
# Chunk sizes are chosen conservatively well below that limit.
_PRICE_CHUNK = 5_000  # 7 params/row → max 9,362; use 5,000
_LABEL_CHUNK = 5_000  # 4 params/row → max 16,383; use 5,000


def _chunks(lst: list, size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def sync_registry(session: Session, instruments: list[Instrument]) -> None:
    """Upsert instruments, groups, and symbol_map rows from the YAML registry.

    Instruments present in the DB but absent from *instruments* are marked
    active=False; their prices are preserved.  Groups and symbol_map rows are
    replaced wholesale on each sync (no price data depends on them directly).
    """
    yaml_ids = {inst.id for inst in instruments}

    existing = session.execute(select(InstrumentRow)).scalars().all()
    for row in existing:
        if row.id not in yaml_ids:
            row.active = False

    for inst in instruments:
        stmt = (
            pg_insert(InstrumentRow)
            .values(
                id=inst.id,
                name=inst.name,
                asset_class=inst.asset_class.value,
                units=inst.units.value,
                quote=inst.quote,
                native_frequency=inst.native_frequency.value,
                valid_from=_parse_date(inst.valid_from),
                valid_to=_parse_date(inst.valid_to),
                notes=inst.notes,
                active=True,
            )
            .on_conflict_do_update(
                index_elements=["id"],
                set_={
                    "name": pg_insert(InstrumentRow).excluded.name,
                    "asset_class": pg_insert(InstrumentRow).excluded.asset_class,
                    "units": pg_insert(InstrumentRow).excluded.units,
                    "quote": pg_insert(InstrumentRow).excluded.quote,
                    "native_frequency": pg_insert(InstrumentRow).excluded.native_frequency,
                    "valid_from": pg_insert(InstrumentRow).excluded.valid_from,
                    "valid_to": pg_insert(InstrumentRow).excluded.valid_to,
                    "notes": pg_insert(InstrumentRow).excluded.notes,
                    "active": True,
                },
            )
        )
        session.execute(stmt)

        session.execute(
            delete(InstrumentGroupRow).where(InstrumentGroupRow.instrument_id == inst.id)
        )
        for group in inst.groups:
            session.add(InstrumentGroupRow(instrument_id=inst.id, group=group))

        session.execute(delete(SymbolMapRow).where(SymbolMapRow.instrument_id == inst.id))
        for src in inst.sources:
            session.add(
                SymbolMapRow(
                    instrument_id=inst.id,
                    provider=src.provider,
                    symbol=src.symbol,
                    priority=src.priority,
                    fidelity=src.fidelity.value,
                    source_frequency=src.source_frequency.value if src.source_frequency else None,
                    publication_lag_days=src.publication_lag_days,
                    revised=src.revised,
                    notes=src.notes,
                )
            )


def upsert_prices(
    session: Session,
    rows: list[dict],
    ingested_at: Optional[datetime] = None,
) -> int:
    """Batch-upsert price rows; return the number of rows inserted or changed.

    A row is only updated when *value* or *known_at* IS DISTINCT FROM the stored
    value (NULL-safe comparison), so a repeated identical load returns 0 and
    the original *ingested_at* is preserved.

    Large batches are split into chunks of at most _PRICE_CHUNK rows to stay
    under Postgres's 65,535 bind-parameter limit.

    Each dict must contain: instrument_id, provider, obs_date, value
    (Decimal or float), known_at (date), and optionally run_id.
    """
    if not rows:
        return 0
    ts = ingested_at or datetime.now(timezone.utc)
    total = 0
    for chunk in _chunks(rows, _PRICE_CHUNK):
        payload = [{**r, "ingested_at": ts} for r in chunk]
        ins = pg_insert(PriceRow.__table__).values(payload)
        stmt = ins.on_conflict_do_update(
            index_elements=["instrument_id", "provider", "obs_date"],
            set_={
                "value": ins.excluded.value,
                "known_at": ins.excluded.known_at,
                "ingested_at": ins.excluded.ingested_at,
                "run_id": ins.excluded.run_id,
            },
            where=or_(
                PriceRow.__table__.c.value.is_distinct_from(ins.excluded.value),
                PriceRow.__table__.c.known_at.is_distinct_from(ins.excluded.known_at),
            ),
        ).returning(PriceRow.__table__.c.id)
        result = session.execute(stmt)
        total += len(result.fetchall())
    return total


def upsert_labels(session: Session, rows: list[dict]) -> int:
    """Batch-upsert label rows; return the number of rows inserted or changed.

    Only updates when *value* or *source* IS DISTINCT FROM the stored row.

    Each dict must contain: label_set (str), week (date), value (int), source (str).
    """
    if not rows:
        return 0
    total = 0
    for chunk in _chunks(rows, _LABEL_CHUNK):
        ins = pg_insert(LabelRow.__table__).values(chunk)
        stmt = ins.on_conflict_do_update(
            index_elements=["label_set", "week"],
            set_={
                "value": ins.excluded.value,
                "source": ins.excluded.source,
            },
            where=or_(
                LabelRow.__table__.c.value.is_distinct_from(ins.excluded.value),
                LabelRow.__table__.c.source.is_distinct_from(ins.excluded.source),
            ),
        ).returning(LabelRow.__table__.c.label_set)
        result = session.execute(stmt)
        total += len(result.fetchall())
    return total


def start_run(session: Session, mode: str) -> IngestionRunRow:
    """Insert a new ingestion_runs row with status='running' and return it."""
    run = IngestionRunRow(
        started_at=datetime.now(timezone.utc),
        mode=mode,
        status="running",
    )
    session.add(run)
    session.flush()
    return run


def finish_run(
    session: Session,
    run: IngestionRunRow,
    status: str,
    instruments_ok: int,
    instruments_failed: int,
    rows_written: int,
) -> None:
    """Update an existing ingestion_runs row to its final state."""
    run.finished_at = datetime.now(timezone.utc)
    run.status = status
    run.instruments_ok = instruments_ok
    run.instruments_failed = instruments_failed
    run.rows_written = rows_written


def add_run_item(
    session: Session,
    run: IngestionRunRow,
    instrument_id: str,
    provider: str,
    status: str,
    rows_written: int = 0,
    first_date: Optional[date] = None,
    last_date: Optional[date] = None,
    error: Optional[str] = None,
) -> IngestionRunItemRow:
    """Append a per-instrument outcome row to an ingestion run."""
    item = IngestionRunItemRow(
        run_id=run.id,
        instrument_id=instrument_id,
        provider=provider,
        status=status,
        rows_written=rows_written,
        first_date=first_date,
        last_date=last_date,
        error=error,
    )
    session.add(item)
    return item


def get_last_price_date(
    session: Session,
    instrument_id: str,
    provider: str,
) -> Optional[date]:
    """Return the most recent obs_date for this instrument/provider, or None."""
    return session.execute(
        select(PriceRow.obs_date)
        .where(
            PriceRow.instrument_id == instrument_id,
            PriceRow.provider == provider,
        )
        .order_by(PriceRow.obs_date.desc())
        .limit(1)
    ).scalar_one_or_none()


# ── Internal helpers ──────────────────────────────────────────────────────────


def _parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    return date.fromisoformat(s)
