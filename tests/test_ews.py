"""Tests for EwsCsvProvider (pure) and load_ews + parity (DB)."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from market_anomaly_detector.data import FEATURE_COLS, engineer_features, load_data
from market_anomaly_detector.paths import DATA_PATH
from market_anomaly_detector.providers.ews import EwsCsvProvider, load_ews, rebuild_ews_frame

# ── EwsCsvProvider unit tests (no DB) ─────────────────────────────────────────

_PROVIDER = EwsCsvProvider()  # shared instance; CSV is read once


def test_series_columns_count():
    assert len(_PROVIDER.series_columns()) == 42


def test_series_columns_excludes_y_and_data():
    cols = _PROVIDER.series_columns()
    assert "Y" not in cols
    assert "Data" not in cols


def test_series_columns_spot_check():
    cols = _PROVIDER.series_columns()
    for expected in ("XAU BGNL", "VIX", "DXY", "GT10", "MXCN"):
        assert expected in cols, f"{expected!r} missing from series_columns()"


def test_get_history_returns_date_objects():
    hist = _PROVIDER.get_history("VIX")
    assert not hist.empty
    assert isinstance(hist["obs_date"].iloc[0], date)


def test_get_history_sorted_ascending():
    hist = _PROVIDER.get_history("DXY")
    assert hist["obs_date"].is_monotonic_increasing


def test_get_history_no_nans():
    hist = _PROVIDER.get_history("VIX")
    assert hist["value"].notna().all()


def test_get_history_first_date():
    hist = _PROVIDER.get_history("XAU BGNL")
    assert hist["obs_date"].iloc[0] == date(2000, 1, 11)


def test_get_history_row_count():
    hist = _PROVIDER.get_history("DXY")
    assert len(hist) == 1111


def test_get_history_unknown_symbol_raises():
    with pytest.raises(KeyError, match="NOTASYMBOL"):
        _PROVIDER.get_history("NOTASYMBOL")


def test_get_labels_no_nans():
    labels = _PROVIDER.get_labels()
    assert labels["value"].notna().all()


def test_get_labels_binary():
    labels = _PROVIDER.get_labels()
    assert set(labels["value"].unique()) <= {0, 1}


def test_get_labels_date_objects():
    labels = _PROVIDER.get_labels()
    assert isinstance(labels["week"].iloc[0], date)


def test_get_labels_row_count():
    labels = _PROVIDER.get_labels()
    assert len(labels) == 1111


# ── load_ews (requires DB) ────────────────────────────────────────────────────


@pytest.mark.db
def test_load_ews_returns_counts(db_session):
    """load_ews must report 46 662 prices (42 × 1111) and 1111 labels written."""
    result = load_ews(db_session)
    db_session.flush()

    assert result["instruments_failed"] == 0
    assert result["instruments_ok"] == 42
    assert result["prices_written"] == 42 * 1111
    assert result["labels_written"] == 1111


@pytest.mark.db
def test_load_ews_idempotent(db_session):
    """A second load_ews call on identical data must write 0 rows."""
    load_ews(db_session)
    db_session.flush()

    result2 = load_ews(db_session)
    db_session.flush()

    assert result2["prices_written"] == 0
    assert result2["labels_written"] == 0


@pytest.mark.db
def test_load_ews_savepoint_isolates_error(db_session, monkeypatch):
    """A DB overflow in one series rolls back only that series; the other 41 load."""
    from sqlalchemy import func, select

    from market_anomaly_detector.db.models import PriceRow
    from market_anomaly_detector.registry import ews_symbol_map, load_registry

    bad_symbol = "VIX"
    original = EwsCsvProvider.get_history

    def _inject_overflow(self, symbol, start=None, end=None):
        hist = original(self, symbol, start=start, end=end)
        if symbol == bad_symbol:
            hist = hist.copy()
            hist["value"] = 1e15  # overflows Numeric(20, 8): needs 16 integer digits
        return hist

    monkeypatch.setattr(EwsCsvProvider, "get_history", _inject_overflow)

    result = load_ews(db_session)
    db_session.flush()

    assert result["instruments_failed"] == 1
    assert result["instruments_ok"] == 41

    sym_map = ews_symbol_map(load_registry())
    bad_id = sym_map[bad_symbol]

    bad_count = db_session.execute(
        select(func.count())
        .select_from(PriceRow)
        .where(PriceRow.instrument_id == bad_id, PriceRow.provider == "bloomberg_ews")
    ).scalar_one()
    assert bad_count == 0, "Overflowed series must have no rows in the DB"

    total_count = db_session.execute(
        select(func.count()).select_from(PriceRow).where(PriceRow.provider == "bloomberg_ews")
    ).scalar_one()
    assert total_count == 41 * 1111, "The other 41 series must load completely"


# ── Exact parity (requires DB) ────────────────────────────────────────────────


@pytest.mark.db
def test_ews_parity_exact(db_session):
    """Exact Decimal-level parity between the DB and the EWS CSV.

    Prices: Decimal(csv_text) == stored Decimal for every (date, series) pair.
    Labels: int(csv_Y) == stored int for every week.
    No rounding — the comparison uses Python Decimal equality which ignores
    trailing zeros (Decimal('283.250') == Decimal('283.25000000')).
    """
    from sqlalchemy import select

    from market_anomaly_detector.db.models import LabelRow, PriceRow
    from market_anomaly_detector.registry import ews_symbol_map, load_registry

    instruments = load_registry()
    load_ews(db_session, instruments=instruments)
    db_session.flush()

    # ── CSV reference (read as strings to preserve exact text) ──────────────
    raw_str = pd.read_csv(DATA_PATH, dtype=str)
    raw_str["obs_date"] = pd.to_datetime(raw_str["Data"], format="%m/%d/%Y").dt.date
    ews_cols = _PROVIDER.series_columns()

    sym_map = ews_symbol_map(instruments)  # {ews_symbol: instrument_id}

    # {(obs_date, instrument_id): Decimal(csv_text)}
    long = raw_str[["obs_date"] + ews_cols].melt(
        id_vars=["obs_date"], var_name="symbol", value_name="val_str"
    )
    csv_prices: dict[tuple, Decimal] = {
        (row.obs_date, sym_map[row.symbol]): Decimal(row.val_str.strip())
        for row in long.itertuples(index=False)
    }

    # ── DB prices ────────────────────────────────────────────────────────────
    db_rows = db_session.execute(
        select(PriceRow.obs_date, PriceRow.instrument_id, PriceRow.value).where(
            PriceRow.provider == "bloomberg_ews"
        )
    ).all()
    db_prices: dict[tuple, Decimal] = {(r.obs_date, r.instrument_id): r.value for r in db_rows}

    assert db_prices.keys() == csv_prices.keys(), (
        f"Key mismatch: {len(db_prices)} DB rows vs {len(csv_prices)} CSV rows"
    )
    mismatches = {
        k: (csv_prices[k], db_prices[k]) for k in csv_prices if csv_prices[k] != db_prices[k]
    }
    assert not mismatches, (
        f"{len(mismatches)} value mismatches (first 3): {dict(list(mismatches.items())[:3])}"
    )

    # ── Labels ────────────────────────────────────────────────────────────────
    csv_labels: dict[date, int] = {
        row.obs_date: int(row.Y.strip())
        for row in raw_str[["obs_date", "Y"]].itertuples(index=False)
    }
    db_label_rows = db_session.execute(
        select(LabelRow.week, LabelRow.value).where(LabelRow.label_set == "ews_v1")
    ).all()
    db_labels: dict[date, int] = {r.week: r.value for r in db_label_rows}

    assert db_labels.keys() == csv_labels.keys()
    assert db_labels == csv_labels


# ── Training/backtest parity (requires DB) ────────────────────────────────────


@pytest.mark.db
def test_db_frame_produces_identical_pipeline_predictions(db_session, trained_pipeline):
    """A frame rebuilt from the DB must yield bit-for-bit identical predictions.

    1. load_ews populates the DB.
    2. rebuild_ews_frame produces a DataFrame equivalent to load_data().
    3. engineer_features on the DB frame equals engineer_features on the CSV frame.
    4. A pipeline trained on the DB frame produces the same calibrated probabilities
       and class labels on the held-out test split as the CSV-trained reference
       pipeline — proving that training and walk-forward backtest results are
       identical when the data source switches from CSV to database.
    """
    from market_anomaly_detector.pipeline import AnomalyPipeline
    from market_anomaly_detector.splits import split_timeseries

    ref_pipeline, _val, ref_test = trained_pipeline

    # Populate DB and rebuild the raw frame
    load_ews(db_session)
    db_session.flush()
    db_raw = rebuild_ews_frame(db_session)

    # Featured frames must be identical (proves all downstream results match)
    csv_raw = load_data(str(DATA_PATH))
    csv_featured = engineer_features(csv_raw)
    db_featured = engineer_features(db_raw)

    pd.testing.assert_frame_equal(
        db_featured.reset_index(drop=True),
        csv_featured.reset_index(drop=True),
        check_exact=True,
        check_names=True,
    )

    # Train from the DB frame and compare predictions on the test split
    db_train, db_val, db_test = split_timeseries(db_featured)
    db_pipeline = AnomalyPipeline()
    db_pipeline.fit(db_train[FEATURE_COLS], db_train["Y"], db_val[FEATURE_COLS], db_val["Y"])

    np.testing.assert_array_equal(
        db_pipeline.predict(db_test[FEATURE_COLS]),
        ref_pipeline.predict(ref_test[FEATURE_COLS]),
    )
    np.testing.assert_array_equal(
        db_pipeline.predict_proba(db_test[FEATURE_COLS])[:, 1],
        ref_pipeline.predict_proba(ref_test[FEATURE_COLS])[:, 1],
    )
