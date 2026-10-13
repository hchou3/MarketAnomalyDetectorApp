"""Shared test fixtures — session-scoped so the pipeline trains once per run."""
import pytest

from market_anomaly_detector.data import engineer_features, load_data, FEATURE_COLS
from market_anomaly_detector.paths import DATA_PATH
from market_anomaly_detector.pipeline import AnomalyPipeline
from market_anomaly_detector.splits import split_timeseries


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
