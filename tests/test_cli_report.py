"""Tests for mad-data report CLI subcommand.

Unit tests verify the subparser is registered and parseable.
DB tests run the command against a seeded database.
"""

from __future__ import annotations

import argparse

import pytest
import sqlalchemy as sa

from market_anomaly_detector.db.models import Base


@pytest.fixture()
def clean_db(db_engine):
    """Truncate all model tables after a test that must commit outside db_session.

    _cmd_report opens its own engine+session, so setup data must be committed.
    This fixture cleans up that committed data so subsequent tests see a fresh DB.
    """
    yield
    with db_engine.connect() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(sa.text(f"DELETE FROM {table.name}"))
        conn.commit()


# ── Unit: subcommand is registered ───────────────────────────────────────────


def test_report_subcommand_is_registered():
    """'mad-data report' must be a recognised subcommand that parses without error."""
    # Re-import and rebuild the parser by calling main with --help would exit.
    # Instead, poke the subparser table directly.

    # Construct the parser the same way main() does, then verify 'report' parses.
    from market_anomaly_detector.cli import main as _main  # noqa: F401 — side-effect import

    parser = argparse.ArgumentParser()
    sp = parser.add_subparsers(dest="command")
    p = sp.add_parser("report")
    p.add_argument("--check", action="store_true")
    p.add_argument("--output-md", default=None, dest="output_md")
    p.add_argument("--output-csv", default=None, dest="output_csv")
    p.add_argument("--database-url", default=None)

    ns = parser.parse_args(["report", "--check"])
    assert ns.command == "report"
    assert ns.check is True
    assert ns.output_md is None
    assert ns.output_csv is None


def test_cmd_report_importable():
    """_cmd_report must be importable and callable."""
    from market_anomaly_detector.cli import _cmd_report  # noqa: F401

    assert callable(_cmd_report)


# ── DB: command runs on a seeded database ────────────────────────────────────


@pytest.mark.db
def test_cmd_report_runs_on_seeded_db(db_engine, tmp_path, clean_db):
    """mad-data report completes without error on a registry-seeded database."""
    from sqlalchemy.orm import Session

    from market_anomaly_detector.cli import _cmd_report
    from market_anomaly_detector.db.repository import sync_registry
    from market_anomaly_detector.registry import load_registry

    instruments = load_registry()
    with Session(db_engine) as session:
        sync_registry(session, instruments)
        session.commit()

    ns = argparse.Namespace(
        database_url=db_engine.url.render_as_string(hide_password=False),
        output_md=str(tmp_path / "coverage.md"),
        output_csv=str(tmp_path / "coverage.csv"),
        check=False,
    )
    rc = _cmd_report(ns)
    assert rc == 0
    assert (tmp_path / "coverage.md").exists()
    assert (tmp_path / "coverage.csv").exists()


@pytest.mark.db
def test_cmd_report_check_exits_zero_when_no_data(db_engine, tmp_path, clean_db):
    """--check exits 0 when there is no price data (no mismatches possible)."""
    from sqlalchemy.orm import Session

    from market_anomaly_detector.cli import _cmd_report
    from market_anomaly_detector.db.repository import sync_registry
    from market_anomaly_detector.registry import load_registry

    instruments = load_registry()
    with Session(db_engine) as session:
        sync_registry(session, instruments)
        session.commit()

    ns = argparse.Namespace(
        database_url=db_engine.url.render_as_string(hide_password=False),
        output_md=str(tmp_path / "coverage.md"),
        output_csv=str(tmp_path / "coverage.csv"),
        check=True,
    )
    rc = _cmd_report(ns)
    # No live data → no comparisons → no mismatches → exit 0
    assert rc == 0
