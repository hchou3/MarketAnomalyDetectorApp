"""Tests for db.known_at (pure) and db.repository (requires Postgres)."""

from datetime import date
from decimal import Decimal

import pytest

from market_anomaly_detector.db.known_at import compute_known_at, obs_period_end
from market_anomaly_detector.db.models import (
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


# ── Repository (require Postgres) ────────────────────────────────────────────

pytestmark_db = pytest.mark.db

# Two minimal instruments used across all DB tests.
_INST_A = {
    "id": "test_gold",
    "instrument_id": "test_gold",
}
_INST_B = {
    "id": "test_vix",
    "instrument_id": "test_vix",
}


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


@pytest.mark.db
def test_sync_registry_inserts_instruments(db_session):
    instruments = load_registry()
    sync_registry(db_session, instruments)
    db_session.flush()

    result = db_session.get(InstrumentRow, "gold")
    assert result is not None
    assert result.name == "Gold"
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

    # Re-sync with one instrument removed.
    subset = [i for i in instruments if i.id != "gold"]
    sync_registry(db_session, subset)
    db_session.flush()

    gold = db_session.get(InstrumentRow, "gold")
    assert gold is not None, "Row should be kept (not deleted)"
    assert gold.active is False


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
    upsert_prices(db_session, [{**base, "value": Decimal("100.0")}])
    db_session.flush()
    upsert_prices(db_session, [{**base, "value": Decimal("200.0")}])
    db_session.flush()

    from sqlalchemy import select

    price = db_session.execute(
        select(PriceRow).where(
            PriceRow.instrument_id == "update_test",
            PriceRow.provider == "fred",
            PriceRow.obs_date == date(2023, 2, 1),
        )
    ).scalar_one()
    assert price.value == Decimal("200.0")


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
