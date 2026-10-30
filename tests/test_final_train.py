"""Tests for final_train.py — final production fit, ONNX export, and model registry."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

# ── helpers ────────────────────────────────────────────────────────────────────


def _make_synthetic_data(
    n_weeks: int = 200, n_features: int = 51, nan_rate: float = 0.15, seed: int = 42
):
    """Synthetic feature DataFrame and labels DataFrame for CI tests."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_weeks, n_features)).astype(float)
    nan_mask = rng.random(X.shape) < nan_rate
    X[nan_mask] = np.nan

    base = date(2010, 1, 5)  # arbitrary Tuesday
    weeks = [base + timedelta(weeks=i) for i in range(n_weeks)]
    # ~15% positive rate
    values = (rng.random(n_weeks) < 0.15).astype(int)

    col_names = [f"feat_{i:02d}" for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=col_names)
    labels_df = pd.DataFrame({"week": weeks, "value": values})
    return X_df, labels_df, weeks


def _make_long_synthetic_data(
    n_weeks: int = 1065,
    n_features: int = 51,
    nan_rate: float = 0.05,
    seed: int = 7,
    val_start: date = date(2019, 4, 23),
    val_end: date = date(2021, 4, 20),
):
    """Synthetic data covering the production validation window."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_weeks, n_features)).astype(float)
    nan_mask = rng.random(X.shape) < nan_rate
    X[nan_mask] = np.nan

    base = date(2001, 1, 2)  # Tuesday far before val_start
    weeks = [base + timedelta(weeks=i) for i in range(n_weeks)]

    # Ensure ≥ 8 positives in val window
    values = np.zeros(n_weeks, dtype=int)
    for i, w in enumerate(weeks):
        if val_start <= w <= val_end:
            values[i] = int(rng.random() < 0.20)
    # Guarantee at least 8 positives in val window regardless of RNG
    val_idx = [i for i, w in enumerate(weeks) if val_start <= w <= val_end]
    if sum(values[i] for i in val_idx) < 8:
        for i in val_idx[:10]:
            values[i] = 1
    # Also some positives in training set
    for i, w in enumerate(weeks):
        if w < val_start and rng.random() < 0.12:
            values[i] = 1

    col_names = [f"feat_{j:02d}" for j in range(n_features)]
    X_df = pd.DataFrame(X, columns=col_names)
    labels_df = pd.DataFrame({"week": weeks, "value": values})
    return X_df, labels_df


# ── spec hash ──────────────────────────────────────────────────────────────────


def test_spec_hash_matches_pinned():
    """PINNED_SPEC_HASH must match the current SPEC_V1 content."""
    from market_anomaly_detector.features.spec import SPEC_V1, spec_hash
    from market_anomaly_detector.final_train import PINNED_SPEC_HASH

    assert spec_hash(SPEC_V1) == PINNED_SPEC_HASH


# ── ONNX parity ────────────────────────────────────────────────────────────────


def test_onnx_parity_synthetic():
    """HGB + Platt -> ONNX max_diff ≤ 1e-5 on synthetic data with NaN features."""
    pytest.importorskip("onnxruntime")
    pytest.importorskip("skl2onnx")

    import numpy as np

    from market_anomaly_detector.calibration import PlattCalibrator
    from market_anomaly_detector.selection import check_onnx_gate, make_hgb_pipeline

    n_samples, n_features = 200, 51
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n_samples, n_features)).astype(float)
    nan_mask = rng.random(X.shape) < 0.15
    X[nan_mask] = np.nan
    y = (rng.random(n_samples) < 0.15).astype(int)

    # Need at least one positive and one negative for Platt
    y[:5] = 1
    y[5:10] = 0

    n_train = 140
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    pipe = make_hgb_pipeline()
    pipe.fit(X_train, y_train)

    raw_val = pipe.decision_function(X_val)
    platt = PlattCalibrator(use_logit=False).fit(raw_val, y_val)

    gate_X = X_train[:100].astype(np.float32)
    gate = check_onnx_gate("hgb", pipe, platt, gate_X, n_features)

    assert gate.passed, f"ONNX parity check failed: max_diff={gate.max_diff}, error={gate.error}"
    assert gate.max_diff <= 1e-5


# ── metadata fields ────────────────────────────────────────────────────────────


def test_metadata_has_required_fields(tmp_path):
    """_fit_final_model writes a metadata.json with all required fields."""
    import json

    pytest.importorskip("onnxruntime")
    pytest.importorskip("skl2onnx")

    from market_anomaly_detector.final_train import _fit_final_model

    val_start = date(2019, 4, 23)
    val_end = date(2021, 4, 20)

    X_df, labels_df = _make_long_synthetic_data(val_start=val_start, val_end=val_end)

    model_version, artifact_dir = _fit_final_model(
        X_df,
        labels_df,
        out_dir=tmp_path,
        label_set="test_set",
        val_start=val_start,
        val_end=val_end,
    )

    meta_path = artifact_dir / "metadata.json"
    assert meta_path.exists(), "metadata.json not written"

    meta = json.loads(meta_path.read_text())

    required_keys = [
        "version",
        "created_at",
        "model_type",
        "spec_hash",
        "config_hash",
        "feature_order",
        "feature_labels",
        "label_set",
        "horizon",
        "training_range",
        "val_range",
        "feature_frame_hash",
        "library_versions",
        "platt_a",
        "platt_b",
        "threshold",
        "onnx_parity_max_diff",
        "heads",
        "dev_metrics",
        "confirm_metrics",
    ]
    missing = [k for k in required_keys if k not in meta]
    assert not missing, f"metadata.json missing keys: {missing}"

    assert meta["version"] == model_version
    assert meta["label_set"] == "test_set"
    assert meta["model_type"] == "hgb"
    assert meta["horizon"] == 4

    heads = meta["heads"]
    assert "detect" in heads and "predict" in heads
    assert heads["detect"]["role"] == "serving"
    assert heads["predict"]["role"] == "experimental"

    assert (artifact_dir / "model.joblib").exists()
    assert (artifact_dir / "model.onnx").exists()
    assert (artifact_dir / "train_X.npy").exists()


# ── activate enforces one active ───────────────────────────────────────────────


@pytest.mark.db
def test_activate_enforces_one_active(db_session):
    """run_activate sets active=True on one version and False on all others."""
    from datetime import datetime, timezone

    from market_anomaly_detector.db.models import ModelVersionRow
    from market_anomaly_detector.final_train import run_activate

    now = datetime.now(timezone.utc)

    v1 = ModelVersionRow(
        version="v1-2026-01-01-b8c2d65c",
        created_at=now,
        label_set="ews_v1",
        spec_hash="b8c2d65c",
        config_hash="331acff8",
        heads="{}",
        metrics="{}",
        threshold=0.5,
        platt_a=1.0,
        platt_b=0.0,
        artifact_path="/tmp/v1",
        active=False,
    )
    v2 = ModelVersionRow(
        version="v1-2026-06-01-b8c2d65c",
        created_at=now,
        label_set="ews_v1",
        spec_hash="b8c2d65c",
        config_hash="331acff8",
        heads="{}",
        metrics="{}",
        threshold=0.5,
        platt_a=1.0,
        platt_b=0.0,
        artifact_path="/tmp/v2",
        active=False,
    )
    db_session.add(v1)
    db_session.add(v2)
    db_session.flush()

    rc = run_activate("v1-2026-01-01-b8c2d65c", "detect", db_session)
    assert rc == 0

    db_session.expire_all()
    row1 = db_session.get(ModelVersionRow, "v1-2026-01-01-b8c2d65c")
    row2 = db_session.get(ModelVersionRow, "v1-2026-06-01-b8c2d65c")
    assert row1.active is True
    assert row2.active is False

    # Switching active to v2 must deactivate v1.
    rc2 = run_activate("v1-2026-06-01-b8c2d65c", "detect", db_session)
    assert rc2 == 0

    db_session.expire_all()
    row1b = db_session.get(ModelVersionRow, "v1-2026-01-01-b8c2d65c")
    row2b = db_session.get(ModelVersionRow, "v1-2026-06-01-b8c2d65c")
    assert row1b.active is False
    assert row2b.active is True


@pytest.mark.db
def test_activate_rejects_predict_head(db_session):
    """run_activate refuses to activate the predict head in v1."""
    from market_anomaly_detector.final_train import run_activate

    rc = run_activate("any-version", "predict", db_session)
    assert rc == 1
