"""Shared test fixtures — session-scoped so the pipeline trains once per run."""

import os

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from market_anomaly_detector.data import FEATURE_COLS, engineer_features, load_data
from market_anomaly_detector.db.models import Base
from market_anomaly_detector.paths import DATA_PATH
from market_anomaly_detector.pipeline import AnomalyPipeline
from market_anomaly_detector.splits import split_timeseries

# ── Database fixtures (require MAD_TEST_DATABASE_URL) ────────────────────────


@pytest.fixture(scope="session")
def db_engine():
    url = os.environ.get("MAD_TEST_DATABASE_URL")
    if not url:
        pytest.skip("MAD_TEST_DATABASE_URL not set; skipping @pytest.mark.db tests")
    engine = sa.create_engine(url)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
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
