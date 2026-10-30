"""Shared test fixtures — session-scoped so the pipeline trains once per run."""

import os

import pytest
import sqlalchemy as sa
from alembic.config import Config as AlembicConfig
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from alembic import command as alembic_command
from market_anomaly_detector.data import FEATURE_COLS, engineer_features, load_data
from market_anomaly_detector.db.engine import get_database_url
from market_anomaly_detector.db.models import Base
from market_anomaly_detector.paths import DATA_PATH, REPO_ROOT
from market_anomaly_detector.pipeline import AnomalyPipeline
from market_anomaly_detector.splits import split_timeseries


def pytest_collection_modifyitems(items):
    if os.environ.get("MAD_RUN_NETWORK") == "1":
        return
    skip = pytest.mark.skip(reason="network tests skipped; set MAD_RUN_NETWORK=1 to run")
    for item in items:
        if item.get_closest_marker("network"):
            item.add_marker(skip)


# ── DB safety guards ─────────────────────────────────────────────────────────
#
# These functions are module-level so tests/test_db_guard.py can import and
# unit-test them without spinning up a real database.


def _db_name(url_str: str) -> str:
    """Return the database name component of a SQLAlchemy URL string."""
    return make_url(url_str).database or ""


def _check_test_url(test_url: str, prod_url: str) -> None:
    """Fail loudly if test_url is unsafe for destructive DB tests.

    Two independent checks:
    1. The parsed database names must differ (guards against host/driver tricks).
    2. The test database name must end with '_test'.
    """
    test_db = _db_name(test_url)
    prod_db = _db_name(prod_url)

    if test_db.lower() == prod_db.lower():
        pytest.fail(
            f"MAD_TEST_DATABASE_URL database name ({test_db!r}) matches the app "
            f"database name ({prod_db!r}) — refusing destructive DB tests. "
            "Point MAD_TEST_DATABASE_URL at a separate *_test database."
        )

    if not test_db.lower().endswith("_test"):
        pytest.fail(
            f"MAD_TEST_DATABASE_URL database name ({test_db!r}) does not end "
            "with '_test' — refusing destructive DB tests to prevent accidental "
            "data loss. Rename your test database to end with '_test'."
        )


def _check_live_db(engine: sa.Engine) -> None:
    """Raise RuntimeError if the live connection's database doesn't end with '_test'.

    Called immediately before any DROP in the teardown path so that even if
    the engine was somehow swapped after setup, no destructive DDL runs.
    """
    with engine.connect() as conn:
        live_db: str = conn.execute(sa.text("SELECT current_database()")).scalar() or ""
    if not live_db.lower().endswith("_test"):
        raise RuntimeError(
            f"Live database ({live_db!r}) does not end with '_test' — "
            "refusing drop_all to prevent data loss. "
            "This is a last-resort safety check; do not bypass it."
        )


# ── Database fixtures (require MAD_TEST_DATABASE_URL) ────────────────────────
#
# Run instructions:
#   docker compose up -d db
#   MAD_TEST_DATABASE_URL=postgresql+psycopg://mad:mad@localhost:5432/mad_test \
#       pytest -m db -v
#
# Rules enforced:
#   • MAD_TEST_DATABASE_URL must be set; tests skip otherwise.
#   • The parsed test database name must differ from the app database name
#     (host aliases, driver prefixes, and trailing options cannot bypass this).
#   • The test database name must end with '_test'.
#   • The live connection is re-checked immediately before any DROP.


@pytest.fixture(scope="session")
def db_engine():
    url = os.environ.get("MAD_TEST_DATABASE_URL")
    if not url:
        pytest.skip("MAD_TEST_DATABASE_URL not set; skipping @pytest.mark.db tests")

    prod_url = get_database_url()
    _check_test_url(url, prod_url)

    engine = sa.create_engine(url)

    # Build the schema via alembic so the fixture exercises the migration itself.
    alembic_cfg = AlembicConfig(str(REPO_ROOT / "alembic.ini"))
    alembic_cfg.attributes["database_url"] = url
    alembic_command.upgrade(alembic_cfg, "head")

    yield engine

    # Belt-and-suspenders: re-verify the live database before any destructive DDL.
    try:
        _check_live_db(engine)
        Base.metadata.drop_all(engine)
        with engine.connect() as conn:
            conn.execute(sa.text("DROP TABLE IF EXISTS alembic_version"))
            conn.commit()
    finally:
        engine.dispose()


@pytest.fixture()
def db_session(db_engine):
    # Wrap each test in a connection-level transaction so that even if the code
    # under test calls session.commit(), it only commits to a SAVEPOINT inside
    # the outer transaction.  The outer rollback at the end always cleans up.
    conn = db_engine.connect()
    trans = conn.begin()
    sess = Session(bind=conn, join_transaction_mode="create_savepoint")
    try:
        yield sess
    finally:
        sess.close()
        trans.rollback()
        conn.close()


# ── Model fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def featured_df():
    raw = load_data(str(DATA_PATH))
    return engineer_features(raw)


@pytest.fixture(scope="session")
def trained_pipeline(featured_df):
    train, val, test = split_timeseries(featured_df)
    pipeline = AnomalyPipeline()
    pipeline.fit(train[FEATURE_COLS], train["Y"], val[FEATURE_COLS], val["Y"])
    return pipeline, val, test
