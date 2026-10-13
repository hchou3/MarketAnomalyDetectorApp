"""Smoke tests for AnomalyPipeline — fit, predict, score interface."""
import numpy as np
import pytest
from sklearn.metrics import f1_score

from market_anomaly_detector.data import load_data, FEATURE_COLS
from market_anomaly_detector.paths import DATA_PATH


def test_predict_proba_shape(trained_pipeline):
    pipeline, val, _ = trained_pipeline
    probas = pipeline.predict_proba(val[FEATURE_COLS])
    assert probas.shape == (len(val), 2)


def test_predict_proba_sums_to_one(trained_pipeline):
    pipeline, val, _ = trained_pipeline
    probas = pipeline.predict_proba(val[FEATURE_COLS])
    assert np.allclose(probas.sum(axis=1), 1.0, atol=1e-6)


def test_predict_binary(trained_pipeline):
    pipeline, val, _ = trained_pipeline
    preds = pipeline.predict(val[FEATURE_COLS])
    assert set(preds).issubset({0, 1})


def test_score_output_contract(trained_pipeline):
    """score() must return the documented dict shape."""
    pipeline, _, _ = trained_pipeline
    raw = load_data(str(DATA_PATH))
    result = pipeline.score(raw.iloc[:50])

    assert isinstance(result["anomaly"], bool)
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["top_features"], list)
    assert len(result["top_features"]) > 0
    first = result["top_features"][0]
    assert "feature" in first and "impact" in first


def test_score_raises_on_too_few_rows(trained_pipeline):
    pipeline, _, _ = trained_pipeline
    raw = load_data(str(DATA_PATH))
    with pytest.raises(ValueError):
        pipeline.score(raw.iloc[:5])


def test_f1_beats_random_baseline(trained_pipeline):
    """Ensemble F1 on test set should be meaningfully above 0."""
    pipeline, _, test = trained_pipeline
    preds = pipeline.predict(test[FEATURE_COLS])
    f1 = f1_score(test["Y"], preds)
    assert f1 > 0.3, f"F1 {f1:.4f} is unexpectedly low — check pipeline."
