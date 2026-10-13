"""Shared test fixtures — session-scoped so the pipeline trains once per run."""

import os

import pytest
import sqlalchemy as sa
from alembic.config import Config as AlembicConfig
from sqlalchemy.orm import Session

from alembic import command as alembic_command
from market_anomaly_detector.data import FEATURE_COLS, engineer_features, load_data
from market_anomaly_detector.db.engine import get_database_url
from market_anomaly_detector.db.models import Base
from market_anomaly_detector.paths import DATA_PATH, REPO_ROOT
from market_anomaly_detector.pipeline import AnomalyPipeline
from market_anomaly_detector.splits import split_timeseries

# ── Database fixtures (require MAD_TEST_DATABASE_URL) ────────────────────────
#
# Run instructions:
#   docker compose up -d db
#   MAD_TEST_DATABASE_URL=postgresql+psycopg://mad:mad@localhost:5432/mad_test \
#       pytest -m db -v
#
# MAD_TEST_DATABASE_URL must point to a *separate* database (mad_test, not mad).
# The fixture refuses to run if both env vars are set and point to the same URL.


@pytest.fixture(scope="session")
def db_engine():
    url = os.environ.get("MAD_TEST_DATABASE_URL")
    if not url:
        pytest.skip("MAD_TEST_DATABASE_URL not set; skipping @pytest.mark.db tests")

    prod_url = get_database_url()
    if url.rstrip("/") == prod_url.rstrip("/"):
        pytest.fail(
            "MAD_TEST_DATABASE_URL equals MAD_DATABASE_URL — refusing to run "
            "destructive DB tests against the production database. "
            "Point MAD_TEST_DATABASE_URL at a separate mad_test database."
        )

    engine = sa.create_engine(url)

    # Build the schema via alembic so the fixture exercises the migration itself.
    alembic_cfg = AlembicConfig(str(REPO_ROOT / "alembic.ini"))
    alembic_cfg.attributes["database_url"] = url
    alembic_command.upgrade(alembic_cfg, "head")

    yield engine

    # Tear down: drop all model tables, then alembic_version.
    Base.metadata.drop_all(engine)
    with engine.connect() as conn:
        conn.execute(sa.text("DROP TABLE IF EXISTS alembic_version"))
        conn.commit()
    engine.dispose()


@pytest.fixture()
def db_session(db_engine):
    with Session(db_engine) as sess:
        yield sess
        sess.rollback()


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
