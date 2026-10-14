"""Tests for EwsCsvProvider (pure) and load_ews + parity (DB)."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from market_anomaly_detector.paths import DATA_PATH
from market_anomaly_detector.providers.ews import EwsCsvProvider, load_ews

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


# ── load_ews + parity (requires DB) ───────────────────────────────────────────


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
def test_ews_parity(db_session):
    """Rebuilding the wide weekly frame from the DB must equal the EWS CSV exactly
    (same dates, same 42 columns, values within 0.0005 — half of the file's 3 d.p.)."""
    from sqlalchemy import select

    from market_anomaly_detector.db.models import PriceRow
    from market_anomaly_detector.registry import ews_symbol_map, load_registry

    instruments = load_registry()
    load_ews(db_session, instruments=instruments)
    db_session.flush()

    # ── Reference: wide frame from CSV ──────────────────────────────────────
    provider = EwsCsvProvider()
    ews_cols = provider.series_columns()
    raw = pd.read_csv(DATA_PATH, parse_dates=["Data"])
    raw["_date"] = raw["Data"].dt.date
    ref = raw.set_index("_date")[ews_cols].copy()
    ref.index.name = "obs_date"
    ref = ref.round(3)

    # ── Result: wide frame rebuilt from DB ──────────────────────────────────
    sym_map = ews_symbol_map(instruments)  # {ews_symbol: instrument_id}
    id_to_sym = {v: k for k, v in sym_map.items()}

    db_rows = db_session.execute(
        select(PriceRow.obs_date, PriceRow.instrument_id, PriceRow.value).where(
            PriceRow.provider == "bloomberg_ews"
        )
    ).all()

    db_df = pd.DataFrame(
        [(r.obs_date, id_to_sym[r.instrument_id], float(r.value)) for r in db_rows],
        columns=["obs_date", "symbol", "value"],
    )
    db_wide = db_df.pivot(index="obs_date", columns="symbol", values="value")
    db_wide.columns.name = None
    db_wide = db_wide.round(3)

    # Align columns to CSV order
    db_aligned = db_wide.reindex(index=ref.index, columns=ref.columns)

    pd.testing.assert_frame_equal(
        db_aligned.sort_index().sort_index(axis=1),
        ref.sort_index().sort_index(axis=1),
        check_exact=False,
        rtol=0.0,
        atol=5e-4,
        check_names=True,
    )
