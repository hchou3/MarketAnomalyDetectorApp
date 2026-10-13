"""Tests for SHAPExplainer correctness and shape-handling."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from market_anomaly_detector.data import engineer_features, load_data, FEATURE_COLS
from market_anomaly_detector.explain import SHAPExplainer, _class1_values
from market_anomaly_detector.pipeline import AnomalyPipeline, AUG_FEATURE_COLS
from market_anomaly_detector.splits import split_timeseries

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "FinancialMarketData.xlsx - EWS.csv"


@pytest.fixture(scope="module")
def trained():
    raw = load_data(str(DATA_PATH))
    featured = engineer_features(raw)
    train, val, _ = split_timeseries(featured)
    pipeline = AnomalyPipeline()
    pipeline.fit(train[FEATURE_COLS], train['Y'], val[FEATURE_COLS], val['Y'])
    return pipeline, val


def test_correct_feature_labelling(trained):
    """Each impact in top_features must match direct SHAP class-1 computation."""
    pipeline, val = trained
    X_aug = pipeline._prepare(val[FEATURE_COLS].iloc[:5])

    raw_shap = pipeline.explainer.explainer.shap_values(X_aug, check_additivity=False)

    for row_idx in range(len(X_aug)):
        result = pipeline.explainer.top_features(X_aug.iloc[[row_idx]], n=len(AUG_FEATURE_COLS))

        # Compute reference class-1 values for this row directly
        if isinstance(raw_shap, list):
            ref = np.array(raw_shap[1])[row_idx]
        else:
            arr = np.array(raw_shap)
            if arr.ndim == 3:
                ref = arr[row_idx, :, 1]
            else:
                ref = arr[row_idx]

        ref_by_name = dict(zip(AUG_FEATURE_COLS, ref.tolist()))
        for entry in result:
            expected = ref_by_name[entry["feature"]]
            assert entry["impact"] == pytest.approx(expected, abs=1e-6), (
                f"Row {row_idx}: feature '{entry['feature']}' impact {entry['impact']} "
                f"!= expected {expected}"
            )


def test_unique_feature_names(trained):
    """All feature names in a full top_features call are distinct and equal AUG_FEATURE_COLS."""
    pipeline, val = trained
    X_aug = pipeline._prepare(val[FEATURE_COLS].iloc[[0]])
    result = pipeline.explainer.top_features(X_aug, n=len(AUG_FEATURE_COLS))

    names = [e["feature"] for e in result]
    assert len(names) == len(set(names)), "Duplicate feature names in top_features output"
    assert set(names) == set(AUG_FEATURE_COLS)


def test_length_guard(trained):
    """An explainer with mismatched feature_names raises ValueError."""
    pipeline, val = trained
    X_aug = pipeline._prepare(val[FEATURE_COLS].iloc[[0]])

    bad = SHAPExplainer.__new__(SHAPExplainer)
    bad.explainer = pipeline.explainer.explainer
    bad.feature_names = AUG_FEATURE_COLS[:-1]  # one too few

    with pytest.raises(ValueError, match="does not match"):
        bad.top_features(X_aug)


def test_class1_values_shape_handling():
    """_class1_values correctly extracts class-1 row-0 for all three SHAP shapes."""
    n_feat = 4
    c0 = np.array([[1.0, 2.0, 3.0, 4.0]])
    c1 = np.array([[10.0, 20.0, 30.0, 40.0]])

    # list shape: [class_0, class_1]
    result_list = _class1_values([c0, c1])
    np.testing.assert_array_equal(result_list, c1[0])

    # 2-D shape: (n_rows, n_features)
    result_2d = _class1_values(c1)
    np.testing.assert_array_equal(result_2d, c1[0])

    # 3-D shape: (n_rows, n_features, n_classes)
    arr_3d = np.stack([c0, c1], axis=2)  # shape (1, 4, 2)
    result_3d = _class1_values(arr_3d)
    np.testing.assert_array_equal(result_3d, c1[0])
