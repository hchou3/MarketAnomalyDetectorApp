"""Tests for the time-based train/val/test split — focusing on data leakage."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from market_anomaly_detector.data import engineer_features, load_data
from market_anomaly_detector.splits import split_timeseries

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "FinancialMarketData.xlsx - EWS.csv"


@pytest.fixture(scope="module")
def featured_df():
    raw = load_data(str(DATA_PATH))
    return engineer_features(raw)


def test_no_future_leakage(featured_df):
    """No future dates must appear in earlier splits."""
    train, val, test = split_timeseries(featured_df)

    assert train['Data'].max() < val['Data'].min(), (
        "Last training date must precede first validation date."
    )
    assert val['Data'].max() < test['Data'].min(), (
        "Last validation date must precede first test date."
    )


def test_split_sizes_are_non_empty(featured_df):
    train, val, test = split_timeseries(featured_df)
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0


def test_no_index_overlap(featured_df):
    train, val, test = split_timeseries(featured_df)
    train_dates = set(train['Data'])
    val_dates = set(val['Data'])
    test_dates = set(test['Data'])
    assert train_dates.isdisjoint(val_dates), "Train and val share dates."
    assert val_dates.isdisjoint(test_dates), "Val and test share dates."
    assert train_dates.isdisjoint(test_dates), "Train and test share dates."


def test_splits_are_sorted(featured_df):
    train, val, test = split_timeseries(featured_df)
    for split, name in [(train, "train"), (val, "val"), (test, "test")]:
        assert split['Data'].is_monotonic_increasing, f"{name} split is not sorted by date."
