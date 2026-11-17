"""Tests for score_history.run_score_history (Phase 2 Step 2.8).

All tests are @pytest.mark.db and require MAD_TEST_DATABASE_URL.

Run with:
    docker compose up -d db
    MAD_TEST_DATABASE_URL=postgresql+psycopg://mad:mad@localhost:5432/mad_test \
        pytest -m db -v tests/test_score_history.py
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── shared fixture helpers ────────────────────────────────────────────────────


def _make_artifact(tmp_path: Path, n_features: int = 51) -> tuple[str, Path, object, object]:
    """Train a minimal HGB + Platt artifact with n_features columns.

    Returns (version, artifacts_root, pipe, platt).
    """
    pytest.importorskip("onnxruntime")
    pytest.importorskip("skl2onnx")

    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import Pipeline

    from market_anomaly_detector.calibration import PlattCalibrator, onnx_export_with_platt

    rng = np.random.default_rng(42)
    n_total = 200
    X = rng.standard_normal((n_total, n_features))
    y = (rng.random(n_total) < 0.3).astype(int)
    y[:5] = 1
    y[5:10] = 0

    n_train = 150
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    pipe = Pipeline([("clf", HistGradientBoostingClassifier(max_iter=10, random_state=42))])
    pipe.fit(X_train, y_train)

    platt = PlattCalibrator(use_logit=False)
    platt.fit(pipe.decision_function(X_val), y_val)

    version = "v1-2000-01-01-b8c2d65c"
    artifact_dir = tmp_path / "artifacts" / version
    artifact_dir.mkdir(parents=True)

    joblib.dump(pipe, artifact_dir / "model.joblib")

    from market_anomaly_detector.features.spec import SPEC_V1

    if n_features == 51:
        feature_order = [f.col for f in SPEC_V1]
        feature_labels = {f.col: f.label for f in SPEC_V1}
    else:
        feature_order = [f"feat_{i:02d}" for i in range(n_features)]
        feature_labels = {c: c for c in feature_order}

    onnx_bytes = onnx_export_with_platt(pipe, platt, n_features, X_sample=X_train[:5])
    (artifact_dir / "model.onnx").write_bytes(onnx_bytes)

    metadata = {
        "version": version,
        "feature_order": feature_order,
        "feature_labels": feature_labels,
        "platt_a": float(platt.a),
        "platt_b": float(platt.b),
        "threshold": 0.3,
        "heads": {
            "detect": {
                "role": "serving",
                "platt_a": float(platt.a),
                "platt_b": float(platt.b),
                "threshold": 0.3,
            },
            "predict": {
                "role": "experimental",
                "platt_a": float(platt.a),
                "platt_b": float(platt.b),
                "threshold": 0.3,
            },
        },
    }
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    return version, tmp_path / "artifacts", pipe, platt


def _make_version_row(session, version: str, artifacts_root: Path) -> None:
    """Insert a minimal ModelVersionRow into the DB."""
    from market_anomaly_detector.db.models import ModelVersionRow
    from market_anomaly_detector.final_train import PINNED_CONFIG_HASH, PINNED_SPEC_HASH

    row = ModelVersionRow(
        version=version,
        created_at=datetime.now(timezone.utc),
        label_set="ews_v1",
        spec_hash=PINNED_SPEC_HASH,
        config_hash=PINNED_CONFIG_HASH,
        heads=json.dumps({}),
        metrics=json.dumps({}),
        threshold=0.3,
        platt_a=1.0,
        platt_b=0.0,
        artifact_path=str(artifacts_root / version),
        active=True,
    )
    session.add(row)
    session.flush()


def _make_feature_df(n_weeks: int = 10, seed: int = 0, nan_core_rows: int = 0):
    """Return a feature DataFrame with SPEC_V1 columns.

    If nan_core_rows > 0, the first nan_core_rows rows have NaN in all core
    columns (vix / msci_usa / us_10y_yield instruments) to simulate
    insufficient_data weeks.
    """
    from market_anomaly_detector.features.spec import SPEC_V1

    rng = np.random.default_rng(seed)
    cols = [f.col for f in SPEC_V1]
    X = rng.standard_normal((n_weeks, len(cols)))

    if nan_core_rows > 0:
        core_instruments = {"vix", "msci_usa", "us_10y_yield"}
        core_col_idx = [i for i, f in enumerate(SPEC_V1) if f.instrument_id in core_instruments]
        X[:nan_core_rows, core_col_idx] = np.nan

    base = date(2010, 1, 5)  # Tuesday
    tuesdays = [base + timedelta(weeks=i) for i in range(n_weeks)]
    df = pd.DataFrame(X, columns=cols, index=pd.DatetimeIndex(tuesdays))
    return df, tuesdays


# ── tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.db
def test_score_history_idempotent(db_session, tmp_path):
    """Running score-history twice writes no new rows on the second run."""
    from sqlalchemy import func, select

    from market_anomaly_detector.db.models import RegimeScoreRow
    from market_anomaly_detector.score_history import run_score_history

    version, artifacts_root, _, _ = _make_artifact(tmp_path)
    _make_version_row(db_session, version, artifacts_root)

    feature_df, _ = _make_feature_df(n_weeks=8)

    rc1 = run_score_history(
        db_session, version=version, artifacts_root=artifacts_root, _feature_df=feature_df
    )
    assert rc1 == 0

    count_after_first = db_session.execute(
        select(func.count())
        .select_from(RegimeScoreRow)
        .where(RegimeScoreRow.model_version == version)
    ).scalar()

    rc2 = run_score_history(
        db_session, version=version, artifacts_root=artifacts_root, _feature_df=feature_df
    )
    assert rc2 == 0

    count_after_second = db_session.execute(
        select(func.count())
        .select_from(RegimeScoreRow)
        .where(RegimeScoreRow.model_version == version)
    ).scalar()

    assert count_after_first == count_after_second, (
        f"Second run added rows: {count_after_first} → {count_after_second}"
    )
    assert count_after_first == 8 * 2, "Expected 2 rows per week (detect + predict)"


@pytest.mark.db
def test_insufficient_data_no_probability_or_alert(db_session, tmp_path):
    """Weeks with missing core features get probability=None and alert=None."""
    from sqlalchemy import select

    from market_anomaly_detector.db.models import RegimeScoreRow
    from market_anomaly_detector.score_history import run_score_history

    version, artifacts_root, _, _ = _make_artifact(tmp_path)
    _make_version_row(db_session, version, artifacts_root)

    n_weeks = 10
    n_insuff = 3
    feature_df, tuesdays = _make_feature_df(n_weeks=n_weeks, nan_core_rows=n_insuff)

    rc = run_score_history(
        db_session, version=version, artifacts_root=artifacts_root, _feature_df=feature_df
    )
    assert rc == 0

    insuff_rows = (
        db_session.execute(
            select(RegimeScoreRow).where(
                RegimeScoreRow.model_version == version,
                RegimeScoreRow.head == "detect",
                RegimeScoreRow.status == "insufficient_data",
            )
        )
        .scalars()
        .all()
    )

    assert len(insuff_rows) == n_insuff, f"Expected {n_insuff} insufficient_data rows"
    for row in insuff_rows:
        assert row.probability is None, f"insufficient_data row has probability={row.probability}"
        assert row.alert is None, f"insufficient_data row has alert={row.alert}"


@pytest.mark.db
def test_alert_equals_probability_ge_threshold(db_session, tmp_path):
    """For detect rows with status='ok', alert == (probability >= threshold)."""
    from sqlalchemy import select

    from market_anomaly_detector.db.models import RegimeScoreRow
    from market_anomaly_detector.score_history import run_score_history

    version, artifacts_root, _, _ = _make_artifact(tmp_path)
    _make_version_row(db_session, version, artifacts_root)

    feature_df, _ = _make_feature_df(n_weeks=20, seed=7)

    rc = run_score_history(
        db_session, version=version, artifacts_root=artifacts_root, _feature_df=feature_df
    )
    assert rc == 0

    ok_rows = (
        db_session.execute(
            select(RegimeScoreRow).where(
                RegimeScoreRow.model_version == version,
                RegimeScoreRow.head == "detect",
                RegimeScoreRow.status == "ok",
            )
        )
        .scalars()
        .all()
    )

    assert len(ok_rows) > 0, "No ok detect rows found"
    for row in ok_rows:
        assert row.probability is not None
        expected_alert = row.probability >= row.threshold
        assert row.alert == expected_alert, (
            f"week={row.week}: alert={row.alert} but probability={row.probability:.4f} "
            f">= threshold={row.threshold:.4f} is {expected_alert}"
        )


@pytest.mark.db
def test_predict_head_no_alert(db_session, tmp_path):
    """Predict head rows always have alert=None regardless of probability."""
    from sqlalchemy import select

    from market_anomaly_detector.db.models import RegimeScoreRow
    from market_anomaly_detector.score_history import run_score_history

    version, artifacts_root, _, _ = _make_artifact(tmp_path)
    _make_version_row(db_session, version, artifacts_root)

    feature_df, _ = _make_feature_df(n_weeks=15, seed=3)

    rc = run_score_history(
        db_session, version=version, artifacts_root=artifacts_root, _feature_df=feature_df
    )
    assert rc == 0

    predict_rows = (
        db_session.execute(
            select(RegimeScoreRow).where(
                RegimeScoreRow.model_version == version,
                RegimeScoreRow.head == "predict",
            )
        )
        .scalars()
        .all()
    )

    assert len(predict_rows) == 15, f"Expected 15 predict rows, got {len(predict_rows)}"
    for row in predict_rows:
        assert row.alert is None, (
            f"predict row for week={row.week} has alert={row.alert} (must be None)"
        )
        assert row.status == "experimental", (
            f"predict row has status={row.status!r} (must be 'experimental')"
        )


@pytest.mark.db
def test_onnx_parity_checked_during_scoring(db_session, tmp_path):
    """If model.onnx is corrupt, score-history returns 1 and writes no rows."""
    from sqlalchemy import func, select

    from market_anomaly_detector.db.models import RegimeScoreRow
    from market_anomaly_detector.score_history import run_score_history

    pytest.importorskip("onnxruntime")

    version, artifacts_root, _, _ = _make_artifact(tmp_path)
    _make_version_row(db_session, version, artifacts_root)

    # Replace valid ONNX with garbage bytes
    onnx_path = artifacts_root / version / "model.onnx"
    onnx_path.write_bytes(b"not a valid onnx model")

    feature_df, _ = _make_feature_df(n_weeks=5)

    rc = run_score_history(
        db_session, version=version, artifacts_root=artifacts_root, _feature_df=feature_df
    )
    assert rc == 1, f"Expected return code 1 on corrupt ONNX, got {rc}"

    row_count = db_session.execute(
        select(func.count())
        .select_from(RegimeScoreRow)
        .where(RegimeScoreRow.model_version == version)
    ).scalar()
    assert row_count == 0, f"Expected 0 rows after failed parity check, got {row_count}"
