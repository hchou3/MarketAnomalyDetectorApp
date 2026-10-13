"""
Train the two-stage anomaly pipeline and save all artifacts.

Usage:
    mad-train
    mad-train --data data/raw/ews_weekly_2000_2021.csv
    mad-train --calibration sigmoid
"""

import argparse
import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap as shap_mod
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from .data import FEATURE_COLS, engineer_features, load_data
from .paths import ARTIFACT_DIR, DATA_PATH
from .pipeline import AnomalyPipeline
from .splits import split_timeseries

log = logging.getLogger(__name__)


def _split_metrics(y_true, y_pred) -> dict:
    return {
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def _date_range(df: pd.DataFrame) -> dict:
    return {"start": str(df["Data"].min().date()), "end": str(df["Data"].max().date())}


def train(
    data_path: Path = DATA_PATH,
    artifact_path: Path = ARTIFACT_DIR / "pipeline.joblib",
    calibration: str = "auto",
) -> AnomalyPipeline:
    log.info("Loading data from %s", data_path)
    raw = load_data(str(data_path))
    featured = engineer_features(raw)
    log.info("After feature engineering: %d rows, %d features", len(featured), len(FEATURE_COLS))

    train_df, val_df, test_df = split_timeseries(
        featured, train_years=9, val_years=2, total_years=12
    )
    log.info(
        "Split sizes -- train: %d, val: %d, test: %d\n"
        "  Train: %s to %s\n"
        "  Val:   %s to %s\n"
        "  Test:  %s to %s",
        len(train_df),
        len(val_df),
        len(test_df),
        train_df["Data"].min().date(),
        train_df["Data"].max().date(),
        val_df["Data"].min().date(),
        val_df["Data"].max().date(),
        test_df["Data"].min().date(),
        test_df["Data"].max().date(),
    )

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["Y"]
    X_val = val_df[FEATURE_COLS]
    y_val = val_df["Y"]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df["Y"]

    # ---- Baseline: standalone Isolation Forest (original best single model) ----
    log.info("[Baseline] Isolation Forest (standalone, 3-feature subset as in original notebook)")
    iso_baseline = IsolationForest(n_estimators=500, contamination="auto", random_state=42)
    iso_baseline.fit(X_train[["VIX", "USD_Over_100", "XAU BGNL_cross"]])
    iso_preds = iso_baseline.predict(X_test[["VIX", "USD_Over_100", "XAU BGNL_cross"]])
    iso_preds = np.where(iso_preds == -1, 1, 0)
    log.info("\n%s", classification_report(y_test, iso_preds, digits=4))
    baseline_f1 = f1_score(y_test, iso_preds)
    log.info("Baseline F1 (anomaly class): %.4f", baseline_f1)

    # ---- Two-stage ensemble ----
    log.info("[Phase 1] Training two-stage stacked ensemble (calibration=%r)...", calibration)
    pipeline = AnomalyPipeline(calibration=calibration)
    pipeline.fit(X_train, y_train, X_val, y_val)

    log.info("[Calibration] Method chosen: %s", pipeline.calibration_method_)
    log.info("[Calibration] OOF report (validation split only, selection criterion):")
    for method, scores in pipeline.calibration_report_.items():
        log.info(
            "  %-8s  oof_brier=%.4f  oof_log_loss=%.4f",
            method,
            scores["oof_brier"],
            scores["oof_log_loss"],
        )

    log.info("[Phase 1] Validation performance:")
    val_results = pipeline.evaluate(X_val, y_val, label="Validation")

    log.info("[Phase 1] Test performance:")
    test_results = pipeline.evaluate(X_test, y_test, label="Test")

    ensemble_f1 = test_results["f1"]
    lift = (ensemble_f1 - baseline_f1) / baseline_f1 * 100
    log.info("F1 lift over baseline: %.4f -> %.4f (%+.1f%%)", baseline_f1, ensemble_f1, lift)

    # ---- Test Brier for both calibration methods (reporting only) ----
    log.info("--- (reporting only, not used for selection) ---")
    X_val_aug = pipeline._prepare(X_val)
    X_test_aug = pipeline._prepare(X_test)
    frozen = FrozenEstimator(pipeline._stacking)
    for method in ("sigmoid", "isotonic"):
        cal = CalibratedClassifierCV(frozen, method=method)
        cal.fit(X_val_aug, y_val)
        test_probas = cal.predict_proba(X_test_aug)[:, 1]
        log.info(
            "  %-8s  test_brier=%.4f  test_f1=%.4f",
            method,
            brier_score_loss(y_test, test_probas),
            f1_score(y_test, (test_probas >= 0.5).astype(int)),
        )

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline.save(str(artifact_path))

    # ---- Metadata sidecar ----
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "shap": shap_mod.__version__,
            "joblib": joblib.__version__,
        },
        "splits": {
            "train": _date_range(train_df),
            "val": _date_range(val_df),
            "test": _date_range(test_df),
        },
        "calibration": {
            "chosen_method": pipeline.calibration_method_,
            "report": pipeline.calibration_report_,
        },
        "metrics": {
            "baseline_test_f1": float(baseline_f1),
            "ensemble_val": _split_metrics(y_val, val_results["predictions"]),
            "ensemble_test": _split_metrics(y_test, test_results["predictions"]),
        },
    }
    sidecar = artifact_path.with_suffix(".meta.json")
    sidecar.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("Metadata written to %s", sidecar)

    return pipeline


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Train the Market Anomaly Detector")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Path to EWS CSV")
    parser.add_argument(
        "--out",
        type=Path,
        default=ARTIFACT_DIR / "pipeline.joblib",
        help="Output path for saved pipeline",
    )
    parser.add_argument(
        "--calibration",
        choices=["auto", "sigmoid", "isotonic"],
        default="auto",
        help="Calibration method (default: auto, selects by OOF Brier on the val split)",
    )
    args = parser.parse_args()
    train(args.data, args.out, args.calibration)


if __name__ == "__main__":
    main()
