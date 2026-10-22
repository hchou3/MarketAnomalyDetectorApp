"""Tests for mad-data revalidate and CLI handler smoke tests."""

from __future__ import annotations

import argparse
from datetime import date
from decimal import Decimal

import pytest
import sqlalchemy as sa

from market_anomaly_detector.db.models import Base

# ── DB isolation: committed rows are cleaned up after each test ───────────────


@pytest.fixture()
def clean_db(db_engine):
    yield
    with db_engine.connect() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(sa.text(f"DELETE FROM {table.name}"))
        conn.commit()


# ── DB test: revalidate replaces stale flags ──────────────────────────────────


@pytest.mark.db
def test_cmd_revalidate_replaces_flags(db_engine, clean_db):
    """revalidate must: not flag negative NFCI (allow_negative=True), flag a
    large equity move, and wipe pre-existing stale flags for each source."""
    from sqlalchemy import select as sa_select
    from sqlalchemy.orm import Session

    from market_anomaly_detector.cli import _cmd_revalidate
    from market_anomaly_detector.db.models import (
        DataQualityFlagRow,
        InstrumentRow,
        PriceRow,
    )

    with Session(db_engine) as session:
        # Minimal InstrumentRow entries to satisfy the FK on prices.
        # Validation logic reads these fields from the YAML registry, not the DB.
        session.add(
            InstrumentRow(
                id="nfci",
                name="NFCI",
                asset_class="macro_index",
                units="index_points",
                quote="Z-score",
                native_frequency="weekly",
                active=True,
                notes="",
            )
        )
        session.add(
            InstrumentRow(
                id="msci_usa",
                name="MSCI USA",
                asset_class="equity_index",
                units="index_points",
                quote="Index level",
                native_frequency="daily",
                active=True,
                notes="",
            )
        )
        # NFCI / fred: two negative rows with a 25% relative change.
        # YAML sets allow_negative=True → no negative_value flag.
        # 25% < 50% macro_index threshold → no large_move flag either.
        session.add(
            PriceRow(
                instrument_id="nfci",
                provider="fred",
                obs_date=date(2020, 1, 3),
                value=Decimal("-0.40"),
            )
        )
        session.add(
            PriceRow(
                instrument_id="nfci",
                provider="fred",
                obs_date=date(2020, 1, 10),
                value=Decimal("-0.30"),
            )
        )
        # MSCI USA / yahoo: 20% one-day jump exceeds the 15% equity_index threshold.
        session.add(
            PriceRow(
                instrument_id="msci_usa",
                provider="yahoo",
                obs_date=date(2020, 1, 3),
                value=Decimal("3000.00"),
            )
        )
        session.add(
            PriceRow(
                instrument_id="msci_usa",
                provider="yahoo",
                obs_date=date(2020, 1, 6),
                value=Decimal("3600.00"),
            )
        )
        # Pre-seed a stale flag; revalidate must delete it before writing new ones.
        session.add(
            DataQualityFlagRow(
                instrument_id="msci_usa",
                provider="yahoo",
                obs_date=date(2020, 1, 3),
                rule="negative_value",
                detail="stale",
            )
        )
        session.commit()

    ns = argparse.Namespace(database_url=db_engine.url.render_as_string(hide_password=False))
    rc = _cmd_revalidate(ns)
    assert rc == 0

    with Session(db_engine) as session:
        nfci_flags = (
            session.execute(
                sa_select(DataQualityFlagRow).where(
                    DataQualityFlagRow.instrument_id == "nfci",
                    DataQualityFlagRow.provider == "fred",
                )
            )
            .scalars()
            .all()
        )
        # allow_negative=True in YAML → no negative_value flag for NFCI
        assert all(f.rule != "negative_value" for f in nfci_flags)

        msci_flags = (
            session.execute(
                sa_select(DataQualityFlagRow).where(
                    DataQualityFlagRow.instrument_id == "msci_usa",
                    DataQualityFlagRow.provider == "yahoo",
                )
            )
            .scalars()
            .all()
        )
        rules = {f.rule for f in msci_flags}
        # 20% equity jump must produce a large_move flag
        assert "large_move" in rules
        # Pre-seeded stale flag must be gone (deleted before new flags were written)
        assert not any(f.rule == "negative_value" for f in msci_flags)


# ── Smoke tests: every handler must survive its lazy imports ──────────────────
#
# Each handler uses lazy imports at the top of its function body.  Calling it
# with a minimal Namespace and no live database exercises those imports.
# A bad import (e.g. ImportError: cannot import name 'load_instruments') raises
# before the handler's own try/except and escapes here, failing the test.
# Normal failures (DB unreachable, CSV not found, etc.) are caught inside the
# handler and returned as int exit codes, which the test accepts.

_MAD_DATA_HANDLERS: list[tuple[str, dict]] = [
    ("_cmd_sync_registry", {"allow_symbol_change": False}),
    ("_cmd_load_ews", {"csv": None}),
    ("_cmd_backfill", {"instrument": None, "since": None}),
    ("_cmd_update", {"instrument": None}),
    ("_cmd_status", {"flags": False}),
    ("_cmd_report", {"output_md": None, "output_csv": None, "check": False}),
    ("_cmd_parity_stress", {"csv": None}),
    ("_cmd_revalidate", {}),
    (
        "_cmd_timing_diagnostic",
        {"instruments": None, "output": None, "date": "2024-01-01"},
    ),
]

_MAD_MODEL_HANDLERS: list[tuple[str, dict]] = [
    ("_cmd_episodes", {"label_set": "ews_v1", "merge_gap": 2}),
    (
        "_cmd_baselines",
        {
            "label_set": "ews_v1",
            "horizon": 4,
            "far_budget": 0.05,
            "first_test_year": 2008,
            "last_test_year": 2017,
        },
    ),
    (
        "_cmd_calibration_diag",
        {
            "label_set": "ews_v1",
            "out_dir": None,
            "first_test_year": 2008,
            "last_test_year": 2017,
            "csv": None,
        },
    ),
    (
        "_cmd_export_dev_predictions",
        {
            "label_set": "ews_v1",
            "out": None,
            "first_test_year": 2008,
            "last_test_year": 2017,
            "variant": None,
        },
    ),
    (
        "_cmd_hgb_experiments",
        {
            "label_set": "ews_v1",
            "out_dir": None,
            "no_boot": True,
            "n_boot": None,
            "first_test_year": 2008,
            "last_test_year": 2017,
        },
    ),
    (
        "_cmd_select",
        {
            "label_set": "ews_v1",
            "out_dir": None,
            "no_boot": True,
            "n_boot": None,
            "first_test_year": 2008,
            "last_test_year": 2017,
            "csv": None,
        },
    ),
    (
        "_cmd_confirm",
        {
            "label_set": "ews_v1",
            "csv": None,
            "out_dir": None,
            "no_boot": True,
            "n_boot": None,
            "force": False,
            "reason": "",
        },
    ),
    (
        "_cmd_explain_cases",
        {"label_set": "ews_v1", "csv": None, "out_dir": None},
    ),
]


@pytest.mark.parametrize(
    "handler,ns_kwargs",
    _MAD_DATA_HANDLERS,
    ids=[h for h, _ in _MAD_DATA_HANDLERS],
)
def test_smoke_mad_data_handler_imports(handler, ns_kwargs):
    """mad-data handler must get past its lazy imports without ImportError."""
    import market_anomaly_detector.cli as mod

    func = getattr(mod, handler)
    ns = argparse.Namespace(database_url=None, **ns_kwargs)
    try:
        result = func(ns)
    except SystemExit:
        return  # business-logic exit — imports passed
    assert isinstance(result, int), f"{handler} must return int, got {type(result)}"


@pytest.mark.parametrize(
    "handler,ns_kwargs",
    _MAD_MODEL_HANDLERS,
    ids=[h for h, _ in _MAD_MODEL_HANDLERS],
)
def test_smoke_mad_model_handler_imports(handler, ns_kwargs):
    """mad-model handler must get past its lazy imports without ImportError."""
    import market_anomaly_detector.model_cli as mod

    func = getattr(mod, handler)
    ns = argparse.Namespace(database_url=None, **ns_kwargs)
    try:
        result = func(ns)
    except SystemExit:
        return  # business-logic exit — imports passed
    assert isinstance(result, int), f"{handler} must return int, got {type(result)}"
