"""Repository layer: thin wrappers over SQLAlchemy that own all DB writes.

All mutations go through these functions so the rest of the codebase never
constructs raw INSERT/UPDATE statements.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import NamedTuple, Optional

from sqlalchemy import delete, func, literal_column, or_, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from ..registry import Instrument
from .models import (
    DataQualityFlagRow,
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


class UpsertResult(NamedTuple):
    rows_new: int
    rows_revised: int
    max_abs_revision: Optional[float]
    max_rel_revision: Optional[float]


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
                active=inst.active,
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
                    "active": pg_insert(InstrumentRow).excluded.active,
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
) -> UpsertResult:
    """Batch-upsert price rows; return ``UpsertResult(rows_new, rows_revised, max_abs, max_rel)``.

    A row is only updated when *value* or *known_at* IS DISTINCT FROM the stored
    value (NULL-safe comparison), so a repeated identical load returns (0, 0, None, None)
    and the original *ingested_at* is preserved.

    New inserts are identified by PostgreSQL's ``xmax = 0`` system column trick:
    after ON CONFLICT DO UPDATE, ``xmax = 0`` for genuinely new rows and non-zero
    for rows that were updated (value or known_at changed).

    ``max_abs_revision`` and ``max_rel_revision`` are the largest absolute and
    relative value changes among revised rows; both are None when rows_revised == 0.

    Large batches are split into chunks of at most _PRICE_CHUNK rows to stay
    under Postgres's 65,535 bind-parameter limit.

    Each dict must contain: instrument_id, provider, obs_date, value
    (Decimal or float), known_at (date), and optionally run_id.
    """
    if not rows:
        return UpsertResult(0, 0, None, None)
    ts = ingested_at or datetime.now(timezone.utc)
    total_new = 0
    total_revised = 0

    # Pre-load existing values so we can compute revision magnitude after the upsert.
    inst_provider_pairs = {(r["instrument_id"], r["provider"]) for r in rows}
    old_values: dict[tuple, float] = {}
    for inst_id, prov in inst_provider_pairs:
        batch_dates = [
            r["obs_date"] for r in rows if r["instrument_id"] == inst_id and r["provider"] == prov
        ]
        for ex in session.execute(
            select(PriceRow.obs_date, PriceRow.value).where(
                PriceRow.instrument_id == inst_id,
                PriceRow.provider == prov,
                PriceRow.obs_date.in_(batch_dates),
            )
        ).fetchall():
            old_values[(inst_id, prov, ex.obs_date)] = float(ex.value)

    # xmax = 0 marks rows inserted by this statement; xid compares to an integer
    # directly (it cannot be cast to BIGINT).
    is_new_col = literal_column("(prices.xmax = 0)").label("is_new")
    revised_keys: set[tuple] = set()

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
        ).returning(
            PriceRow.__table__.c.instrument_id,
            PriceRow.__table__.c.provider,
            PriceRow.__table__.c.obs_date,
            is_new_col,
        )
        for returned in session.execute(stmt).fetchall():
            if returned.is_new:
                total_new += 1
            else:
                total_revised += 1
                revised_keys.add((returned.instrument_id, returned.provider, returned.obs_date))

    # Compute revision magnitude for actually-revised rows.
    abs_changes: list[float] = []
    rel_changes: list[float] = []
    for r in rows:
        key = (r["instrument_id"], r["provider"], r["obs_date"])
        if key in revised_keys:
            old_v = old_values.get(key)
            if old_v is not None:
                diff = abs(float(r["value"]) - old_v)
                abs_changes.append(diff)
                if old_v != 0:
                    rel_changes.append(diff / abs(old_v))

    max_abs = max(abs_changes) if abs_changes else None
    max_rel = max(rel_changes) if rel_changes else None
    return UpsertResult(total_new, total_revised, max_abs, max_rel)


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


def write_quality_flags(
    session: Session,
    flags: list,  # list[ValidationFlag] — avoid circular import
    *,
    instrument_id: str,
    provider: str,
    run_id: Optional[int] = None,
) -> None:
    """Persist validation flags for a single (instrument, provider) fetch."""
    for flag in flags:
        session.add(
            DataQualityFlagRow(
                run_id=run_id,
                instrument_id=instrument_id,
                provider=provider,
                obs_date=flag.obs_date,
                rule=flag.rule,
                detail=flag.detail,
            )
        )


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
    rows_new: int = 0,
    rows_revised: int = 0,
    first_date: Optional[date] = None,
    last_date: Optional[date] = None,
    error: Optional[str] = None,
) -> IngestionRunItemRow:
    """Append a per-instrument outcome row to an ingestion run.

    When *rows_new* or *rows_revised* are provided, *rows_written* is computed
    as their sum.  Pass *rows_written* alone for backward-compatible callers
    that don't have the split counts.
    """
    if rows_new > 0 or rows_revised > 0:
        total = rows_new + rows_revised
    else:
        total = rows_written
    item = IngestionRunItemRow(
        run_id=run.id,
        instrument_id=instrument_id,
        provider=provider,
        status=status,
        rows_written=total,
        rows_new=rows_new,
        rows_revised=rows_revised,
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


def get_last_run(session: Session) -> Optional[IngestionRunRow]:
    """Return the most recent ingestion run row, or None."""
    return session.execute(
        select(IngestionRunRow).order_by(IngestionRunRow.started_at.desc()).limit(1)
    ).scalar_one_or_none()


def get_price_coverage(session: Session) -> list:
    """Return per-(instrument_id, provider) coverage: first_date, last_date, row_count."""
    return session.execute(
        select(
            PriceRow.instrument_id,
            PriceRow.provider,
            func.min(PriceRow.obs_date).label("first_date"),
            func.max(PriceRow.obs_date).label("last_date"),
            func.count().label("row_count"),
        )
        .group_by(PriceRow.instrument_id, PriceRow.provider)
        .order_by(PriceRow.instrument_id, PriceRow.provider)
    ).fetchall()


def get_flag_counts_by_rule(session: Session) -> list:
    """Return per-rule flag counts across all runs."""
    return session.execute(
        select(
            DataQualityFlagRow.rule,
            func.count().label("count"),
        )
        .group_by(DataQualityFlagRow.rule)
        .order_by(DataQualityFlagRow.rule)
    ).fetchall()


def get_flag_counts_per_instrument(session: Session) -> list:
    """Return per-(instrument_id, rule) flag counts."""
    return session.execute(
        select(
            DataQualityFlagRow.instrument_id,
            DataQualityFlagRow.rule,
            func.count().label("count"),
        )
        .group_by(DataQualityFlagRow.instrument_id, DataQualityFlagRow.rule)
        .order_by(DataQualityFlagRow.instrument_id, DataQualityFlagRow.rule)
    ).fetchall()


def get_last_run_items(session: Session, run_id: int) -> list:
    """Return all IngestionRunItemRow entries for the given run_id."""
    return (
        session.execute(
            select(IngestionRunItemRow)
            .where(IngestionRunItemRow.run_id == run_id)
            .order_by(IngestionRunItemRow.instrument_id, IngestionRunItemRow.provider)
        )
        .scalars()
        .all()
    )


def get_instrument_last_dates(session: Session) -> list:
    """Return max obs_date and native_frequency per active instrument."""
    return session.execute(
        select(
            InstrumentRow.id,
            InstrumentRow.native_frequency,
            func.max(PriceRow.obs_date).label("last_obs_date"),
        )
        .outerjoin(PriceRow, PriceRow.instrument_id == InstrumentRow.id)
        .where(InstrumentRow.active.is_(True))
        .group_by(InstrumentRow.id, InstrumentRow.native_frequency)
        .order_by(InstrumentRow.id)
    ).fetchall()


def get_source_last_dates(session: Session) -> list:
    """Return max obs_date per active (instrument, provider), with source metadata.

    Each row has: id, native_frequency, provider, source_frequency,
    publication_lag_days, last_obs_date.
    """
    return session.execute(
        select(
            InstrumentRow.id,
            InstrumentRow.native_frequency,
            SymbolMapRow.provider,
            SymbolMapRow.source_frequency,
            SymbolMapRow.publication_lag_days,
            func.max(PriceRow.obs_date).label("last_obs_date"),
        )
        .join(SymbolMapRow, SymbolMapRow.instrument_id == InstrumentRow.id)
        .outerjoin(
            PriceRow,
            (PriceRow.instrument_id == InstrumentRow.id)
            & (PriceRow.provider == SymbolMapRow.provider),
        )
        .where(InstrumentRow.active.is_(True))
        .group_by(
            InstrumentRow.id,
            InstrumentRow.native_frequency,
            SymbolMapRow.provider,
            SymbolMapRow.source_frequency,
            SymbolMapRow.publication_lag_days,
        )
        .order_by(InstrumentRow.id, SymbolMapRow.provider)
    ).fetchall()


# ── Internal helpers ──────────────────────────────────────────────────────────


def _parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    return date.fromisoformat(s)
