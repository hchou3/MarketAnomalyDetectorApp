"""Tests for db.known_at (pure) and db.repository (requires Postgres)."""

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from market_anomaly_detector.db.known_at import compute_known_at, obs_period_end
from market_anomaly_detector.db.models import (
    Base,
    IngestionRunRow,
    InstrumentRow,
    LabelRow,
    PriceRow,
    SymbolMapRow,
)
from market_anomaly_detector.db.repository import (
    add_run_item,
    finish_run,
    get_last_price_date,
    start_run,
    sync_registry,
    upsert_labels,
    upsert_prices,
)
from market_anomaly_detector.registry import load_registry

# ── known_at (pure, no DB) ────────────────────────────────────────────────────


def test_obs_period_end_daily():
    d = date(2023, 3, 15)
    assert obs_period_end(d, "daily") == d


def test_obs_period_end_weekly():
    d = date(2023, 3, 14)
    assert obs_period_end(d, "weekly") == d


def test_obs_period_end_none_frequency():
    d = date(2023, 3, 15)
    assert obs_period_end(d, None) == d


def test_obs_period_end_monthly_regular():
    assert obs_period_end(date(2023, 1, 1), "monthly") == date(2023, 1, 31)


def test_obs_period_end_monthly_february_non_leap():
    assert obs_period_end(date(2023, 2, 1), "monthly") == date(2023, 2, 28)


def test_obs_period_end_monthly_february_leap():
    assert obs_period_end(date(2024, 2, 1), "monthly") == date(2024, 2, 29)


def test_obs_period_end_monthly_december():
    assert obs_period_end(date(2023, 12, 1), "monthly") == date(2023, 12, 31)


def test_compute_known_at_daily_zero_lag():
    assert compute_known_at(date(2023, 3, 15), "daily", 0) == date(2023, 3, 15)


def test_compute_known_at_daily_with_lag():
    assert compute_known_at(date(2023, 3, 15), "daily", 3) == date(2023, 3, 18)


def test_compute_known_at_monthly_uses_period_end():
    # FRED obs_date 2023-01-01 = January 2023; period ends 2023-01-31.
    # With 40-day lag: 2023-01-31 + 40 = 2023-03-12
    assert compute_known_at(date(2023, 1, 1), "monthly", 40) == date(2023, 3, 12)


def test_compute_known_at_monthly_not_from_obs_date():
    # Naively adding lag to obs_date (2023-01-01 + 40 = 2023-02-10) is wrong.
    # The correct answer uses the period end (2023-01-31 + 40 = 2023-03-12).
    result = compute_known_at(date(2023, 1, 1), "monthly", 40)
    naive_wrong = date(2023, 2, 10)  # obs_date + 40 days, ignoring period end
    assert result != naive_wrong
    assert result == date(2023, 3, 12)


def test_compute_known_at_weekly():
    assert compute_known_at(date(2023, 3, 10), "weekly", 11) == date(2023, 3, 21)


# ── Helpers for DB tests ──────────────────────────────────────────────────────


def _make_instrument_row(instrument_id: str) -> dict:
    return dict(
        id=instrument_id,
        name=f"Test {instrument_id}",
        asset_class="volatility",
        units="index_points",
        quote="points",
        native_frequency="daily",
        notes="",
        active=True,
    )


def _insert_instrument(session, instrument_id: str) -> InstrumentRow:
    """Insert a minimal InstrumentRow directly (bypasses registry sync)."""
    row = InstrumentRow(**_make_instrument_row(instrument_id))
    session.add(row)
    session.flush()
    return row


# ── Registry sync ─────────────────────────────────────────────────────────────


@pytest.mark.db
def test_sync_registry_inserts_instruments(db_session):
    instruments = load_registry()
    sync_registry(db_session, instruments)
    db_session.flush()

    result = db_session.get(InstrumentRow, "gold")
    assert result is not None
    assert result.name == "Gold spot (Bloomberg BGN)"
    assert result.active is True


@pytest.mark.db
def test_sync_registry_stores_source_frequency(db_session):
    """source_frequency must be persisted in symbol_map for monthly FRED proxies."""
    sync_registry(db_session, load_registry())
    db_session.flush()

    from sqlalchemy import select

    monthly_ids = {"de_10y_yield", "it_10y_yield", "jp_10y_yield"}
    for inst_id in monthly_ids:
        row = db_session.execute(
            select(SymbolMapRow).where(
                SymbolMapRow.instrument_id == inst_id,
                SymbolMapRow.provider == "fred",
            )
        ).scalar_one_or_none()
        assert row is not None, f"No FRED source for {inst_id}"
        assert row.source_frequency == "monthly", (
            f"{inst_id} FRED source_frequency={row.source_frequency!r}, expected 'monthly'"
        )
        assert row.publication_lag_days >= 28


@pytest.mark.db
def test_sync_registry_is_idempotent(db_session):
    instruments = load_registry()
    sync_registry(db_session, instruments)
    db_session.flush()
    sync_registry(db_session, instruments)
    db_session.flush()

    from sqlalchemy import func, select

    count = db_session.execute(select(func.count()).select_from(InstrumentRow)).scalar_one()
    assert count == len(instruments)


@pytest.mark.db
def test_sync_registry_deactivates_removed_instrument(db_session):
    instruments = load_registry()
    sync_registry(db_session, instruments)
    db_session.flush()

    subset = [i for i in instruments if i.id != "gold"]
    sync_registry(db_session, subset)
    db_session.flush()

    gold = db_session.get(InstrumentRow, "gold")
    assert gold is not None, "Row should be kept (not deleted)"
    assert gold.active is False


# ── upsert_prices ─────────────────────────────────────────────────────────────


@pytest.mark.db
def test_upsert_prices_inserts_rows(db_session):
    _insert_instrument(db_session, "upsert_test")
    rows = [
        dict(
            instrument_id="upsert_test",
            provider="fred",
            obs_date=date(2023, 1, 3),
            value=Decimal("1950.50"),
            known_at=date(2023, 1, 4),
            run_id=None,
        )
    ]
    n = upsert_prices(db_session, rows)
    db_session.flush()
    assert n == 1

    from sqlalchemy import select

    price = db_session.execute(
        select(PriceRow).where(
            PriceRow.instrument_id == "upsert_test",
            PriceRow.provider == "fred",
            PriceRow.obs_date == date(2023, 1, 3),
        )
    ).scalar_one()
    assert price.value == Decimal("1950.50")


@pytest.mark.db
def test_upsert_prices_updates_existing(db_session):
    _insert_instrument(db_session, "update_test")
    base = dict(
        instrument_id="update_test",
        provider="fred",
        obs_date=date(2023, 2, 1),
        known_at=date(2023, 2, 2),
        run_id=None,
    )
    n1 = upsert_prices(db_session, [{**base, "value": Decimal("100.0")}])
    db_session.flush()
    assert n1 == 1

    n2 = upsert_prices(db_session, [{**base, "value": Decimal("200.0")}])
    db_session.flush()
    assert n2 == 1  # value changed → row returned by RETURNING

    from sqlalchemy import select

    db_session.expire_all()
    price = db_session.execute(
        select(PriceRow).where(
            PriceRow.instrument_id == "update_test",
            PriceRow.provider == "fred",
            PriceRow.obs_date == date(2023, 2, 1),
        )
    ).scalar_one()
    assert price.value == Decimal("200.0")


@pytest.mark.db
def test_upsert_prices_repeated_load_returns_zero(db_session):
    """Upserting identical rows a second time must return 0 (no changes made)."""
    _insert_instrument(db_session, "repeat_test")
    row = dict(
        instrument_id="repeat_test",
        provider="fred",
        obs_date=date(2023, 3, 1),
        value=Decimal("100.0"),
        known_at=date(2023, 3, 2),
        run_id=None,
    )
    n1 = upsert_prices(db_session, [row])
    db_session.flush()
    assert n1 == 1

    n2 = upsert_prices(db_session, [row])
    db_session.flush()
    assert n2 == 0


@pytest.mark.db
def test_upsert_prices_preserves_ingested_at_on_repeat(db_session):
    """When value and known_at are unchanged, ingested_at must not be overwritten."""
    _insert_instrument(db_session, "ingested_at_test")

    first_ts = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    row = dict(
        instrument_id="ingested_at_test",
        provider="fred",
        obs_date=date(2023, 3, 1),
        value=Decimal("100.0"),
        known_at=date(2023, 3, 2),
        run_id=None,
    )
    upsert_prices(db_session, [row], ingested_at=first_ts)
    db_session.flush()

    second_ts = datetime(2023, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    upsert_prices(db_session, [row], ingested_at=second_ts)
    db_session.flush()

    from sqlalchemy import select

    db_session.expire_all()
    price = db_session.execute(
        select(PriceRow).where(
            PriceRow.instrument_id == "ingested_at_test",
            PriceRow.obs_date == date(2023, 3, 1),
        )
    ).scalar_one()
    assert price.ingested_at == first_ts


@pytest.mark.db
def test_upsert_prices_large_batch(db_session):
    """16,000 rows must be chunked and all inserted without hitting the PG param limit."""
    _insert_instrument(db_session, "large_batch_test")
    db_session.flush()

    start = date(2000, 1, 3)
    n_rows = 16_000
    rows = [
        dict(
            instrument_id="large_batch_test",
            provider="fred",
            obs_date=start + timedelta(days=i),
            value=Decimal(str(i)),
            known_at=start + timedelta(days=i + 1),
            run_id=None,
        )
        for i in range(n_rows)
    ]
    n = upsert_prices(db_session, rows)
    db_session.flush()
    assert n == n_rows


# ── upsert_labels ─────────────────────────────────────────────────────────────


@pytest.mark.db
def test_upsert_labels(db_session):
    rows = [
        dict(label_set="ews_v1_test", week=date(2010, 1, 5), value=1, source="test"),
        dict(label_set="ews_v1_test", week=date(2010, 1, 12), value=0, source="test"),
    ]
    n = upsert_labels(db_session, rows)
    db_session.flush()
    assert n == 2

    from sqlalchemy import select

    label = db_session.execute(
        select(LabelRow).where(
            LabelRow.label_set == "ews_v1_test",
            LabelRow.week == date(2010, 1, 5),
        )
    ).scalar_one()
    assert label.value == 1


@pytest.mark.db
def test_upsert_labels_repeated_load_returns_zero(db_session):
    rows = [dict(label_set="ews_repeat_test", week=date(2010, 2, 2), value=1, source="test")]
    assert upsert_labels(db_session, rows) == 1
    db_session.flush()
    assert upsert_labels(db_session, rows) == 0


# ── Run lifecycle ─────────────────────────────────────────────────────────────


@pytest.mark.db
def test_run_lifecycle(db_session):
    run = start_run(db_session, mode="backfill")
    assert run.id is not None
    assert run.status == "running"

    item = add_run_item(
        db_session,
        run,
        instrument_id="gold",
        provider="fred",
        status="ok",
        rows_written=100,
        first_date=date(2020, 1, 2),
        last_date=date(2020, 12, 31),
    )
    assert item.rows_written == 100

    finish_run(
        db_session, run, status="success", instruments_ok=1, instruments_failed=0, rows_written=100
    )
    db_session.flush()

    refreshed = db_session.get(IngestionRunRow, run.id)
    assert refreshed.status == "success"
    assert refreshed.finished_at is not None
    assert refreshed.instruments_ok == 1


# ── get_last_price_date ───────────────────────────────────────────────────────


@pytest.mark.db
def test_get_last_price_date_empty(db_session):
    _insert_instrument(db_session, "noprices_test")
    result = get_last_price_date(db_session, "noprices_test", "fred")
    assert result is None


@pytest.mark.db
def test_get_last_price_date_returns_latest(db_session):
    _insert_instrument(db_session, "dates_test")
    rows = [
        dict(
            instrument_id="dates_test",
            provider="yahoo",
            obs_date=d,
            value=Decimal("1.0"),
            known_at=d,
            run_id=None,
        )
        for d in [date(2023, 1, 3), date(2023, 1, 10), date(2023, 1, 5)]
    ]
    upsert_prices(db_session, rows)
    db_session.flush()

    latest = get_last_price_date(db_session, "dates_test", "yahoo")
    assert latest == date(2023, 1, 10)


# ── Migration checks ──────────────────────────────────────────────────────────


@pytest.mark.db
def test_migration_is_at_head(db_engine):
    """After alembic upgrade head the DB must be at the expected revision."""
    from alembic.runtime.migration import MigrationContext

    with db_engine.connect() as conn:
        ctx = MigrationContext.configure(conn)
        heads = ctx.get_current_heads()

    assert "a1b2c3d4e5f6" in heads, f"Expected revision a1b2c3d4e5f6 in heads, got {heads!r}"


@pytest.mark.db
def test_no_model_drift(db_engine):
    """All tables and columns defined in models.py must be present in the DB.

    Extra indexes or constraints added by the migration (not in models) are
    intentional and are not flagged.
    """
    from alembic.autogenerate import compare_metadata
    from alembic.runtime.migration import MigrationContext

    with db_engine.connect() as conn:
        ctx = MigrationContext.configure(conn)
        diff = compare_metadata(ctx, Base.metadata)

    # Only care about things the models expect but the DB is missing.
    # "remove_*" means the DB has something extra (e.g. a performance index)
    # which is not drift — the migration intentionally adds things the ORM
    # class doesn't model.
    missing_from_db = [op for op in diff if not str(op[0]).startswith("remove_")]
    assert not missing_from_db, (
        "Model/migration drift — run 'alembic revision --autogenerate' to create a "
        f"new migration. Missing from DB: {missing_from_db}"
    )
