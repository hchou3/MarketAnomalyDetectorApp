"""Tests for the walk-forward backtest — fold boundaries, coverage, leakage, metrics."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from market_anomaly_detector.backtest import (
    confusion_metrics,
    make_folds,
    run_fold,
    summarise,
    year_block_bootstrap,
)
from market_anomaly_detector.data import engineer_features, load_data

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "FinancialMarketData.xlsx - EWS.csv"


@pytest.fixture(scope="module")
def featured():
    return engineer_features(load_data(str(DATA_PATH)))


@pytest.fixture(scope="module")
def folds(featured):
    return make_folds(featured, first_test_year=2008)


# ---------------------------------------------------------------- folds

def test_folds_are_strictly_time_ordered(featured, folds):
    d = featured['Data']
    for f in folds:
        train_end = d.iloc[f.val_lo - 1]
        val_start, val_end = d.iloc[f.val_lo], d.iloc[f.test_lo - 1]
        test_start = d.iloc[f.test_lo]
        assert train_end < val_start <= val_end < test_start, f"leaky boundaries in {f.test_year}"


def test_each_test_window_is_exactly_its_calendar_year(featured, folds):
    years = featured['Data'].dt.year
    for f in folds:
        assert set(years.iloc[f.test_lo:f.test_hi]) == {f.test_year}
        assert (years == f.test_year).sum() == f.test_hi - f.test_lo


def test_test_windows_cover_every_week_once(featured, folds):
    covered = np.concatenate([np.arange(f.test_lo, f.test_hi) for f in folds])
    assert len(covered) == len(set(covered)), "a week is tested twice"
    expected = np.flatnonzero(featured['Data'].dt.year >= 2008)
    assert np.array_equal(np.sort(covered), expected)


def test_calibration_windows_have_enough_of_each_class(featured, folds):
    y = featured['Y'].to_numpy()
    for f in folds:
        v = y[f.val_lo:f.test_lo]
        assert v.sum() >= 8 and (v == 0).sum() >= 8
        assert f.val_positives == v.sum()


def test_val_window_extension_uses_only_the_past(featured):
    # 2015's default 2-year window (2013-2014) has only 2 anomalies, so it must
    # extend backwards (to 2012) and never forwards.
    f2015 = next(f for f in make_folds(featured, 2015, 2015))
    assert f2015.val_start_year < 2013
    assert featured['Data'].iloc[f2015.test_lo - 1].year == 2014


def test_training_window_expands(folds):
    assert all(b.val_lo >= a.val_lo for a, b in zip(folds, folds[1:]))


def test_impossible_window_raises(featured):
    with pytest.raises(ValueError):
        make_folds(featured, first_test_year=2008, min_val_positives=10_000)


# ---------------------------------------------------------------- leakage

def test_future_labels_cannot_change_a_folds_predictions(featured):
    """
    Corrupt every label from the test year onwards. If the fold only ever sees
    the past, its test-year predictions must be bit-for-bit identical.
    """
    fold = make_folds(featured, 2016, 2016)[0]
    clean, _ = run_fold(featured, fold, calibration="sigmoid")

    tampered = featured.copy()
    future = tampered['Data'].dt.year >= 2016
    tampered.loc[future, 'Y'] = 1 - tampered.loc[future, 'Y']
    dirty, _ = run_fold(tampered, fold, calibration="sigmoid")

    np.testing.assert_array_equal(clean['ens_proba'].to_numpy(), dirty['ens_proba'].to_numpy())
    np.testing.assert_array_equal(clean['base_pred'].to_numpy(), dirty['base_pred'].to_numpy())


def test_run_fold_output_shape(featured):
    fold = make_folds(featured, 2021, 2021)[0]
    preds, info = run_fold(featured, fold)
    assert len(preds) == fold.test_hi - fold.test_lo == info['n_test']
    assert preds['ens_proba'].between(0, 1).all()
    assert set(preds['ens_pred']) <= {0, 1} and set(preds['base_pred']) <= {0, 1}
    assert info['calibration_method'] in {"sigmoid", "isotonic"}


# ---------------------------------------------------------------- metrics

def test_confusion_metrics_known_values():
    m = confusion_metrics([1, 1, 0, 0, 1], [1, 0, 1, 0, 1])
    assert (m['tp'], m['fp'], m['fn'], m['tn']) == (2, 1, 1, 1)
    assert m['precision'] == pytest.approx(2 / 3)
    assert m['recall'] == pytest.approx(2 / 3)
    assert m['f1'] == pytest.approx(2 / 3)
    assert m['false_alarm_rate'] == pytest.approx(0.5)


def test_confusion_metrics_handles_no_positives():
    m = confusion_metrics([0, 0, 0], [0, 1, 0])
    assert m['f1'] == 0.0 and m['recall'] == 0.0 and m['fp'] == 1


def _toy_preds(ens_correct: bool) -> pd.DataFrame:
    rows = []
    for yr in range(2000, 2006):
        y = np.array([1, 1, 0, 0])
        rows.append(pd.DataFrame({
            "Data": pd.date_range(f"{yr}-01-01", periods=4, freq="W"),
            "test_year": yr,
            "Y": y,
            "ens_proba": np.where(y == 1, 0.9, 0.1),
            "ens_pred": y if ens_correct else 1 - y,
            "base_score": np.zeros(4),
            "base_pred": 1 - y if ens_correct else y,
        }))
    return pd.concat(rows, ignore_index=True)


def test_bootstrap_detects_a_clear_winner():
    bs = year_block_bootstrap(_toy_preds(ens_correct=True), n_boot=200, seed=1)
    assert bs['p_ensemble_better'] == 1.0
    assert bs['f1_diff_ci95'][0] > 0


def test_summarise_pools_across_years():
    preds = _toy_preds(ens_correct=True)
    infos = [{"test_year": yr, "calibration_method": "sigmoid", "val_start_year": yr - 2,
              "val_positives": 10, "n_train": 100} for yr in range(2000, 2006)]
    summary, per_year = summarise(preds, infos, n_boot=50)
    assert summary['pooled']['ensemble']['f1'] == 1.0
    assert summary['pooled']['baseline_isolation_forest']['f1'] == 0.0
    assert len(per_year) == 6 and per_year['anomalies'].sum() == 12
