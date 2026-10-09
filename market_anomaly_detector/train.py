"""
Train the two-stage anomaly pipeline and save all artifacts.

Usage:
    python -m market_anomaly_detector.train --data "data/FinancialMarketData.xlsx - EWS.csv"
    python -m market_anomaly_detector.train
"""
import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import shap as shap_mod
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, classification_report, f1_score, precision_score, recall_score

from .data import engineer_features, load_data, FEATURE_COLS
from .pipeline import AnomalyPipeline
from .splits import split_timeseries

DEFAULT_DATA = Path(__file__).parent.parent / "data" / "FinancialMarketData.xlsx - EWS.csv"
DEFAULT_ARTIFACT = Path(__file__).parent.parent / "artifacts" / "pipeline.joblib"


def train(
    data_path: Path = DEFAULT_DATA,
    artifact_path: Path = DEFAULT_ARTIFACT,
    calibration: str = "auto",
) -> AnomalyPipeline:
    print(f"Loading data from {data_path}")
    raw = load_data(str(data_path))
    featured = engineer_features(raw)
    print(f"After feature engineering: {len(featured)} rows, {len(FEATURE_COLS)} features")

    train_df, val_df, test_df = split_timeseries(featured, train_years=9, val_years=2, total_years=12)
    print(
        f"Split sizes -- train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}\n"
        f"  Train: {train_df['Data'].min().date()} to {train_df['Data'].max().date()}\n"
        f"  Val:   {val_df['Data'].min().date()} to {val_df['Data'].max().date()}\n"
        f"  Test:  {test_df['Data'].min().date()} to {test_df['Data'].max().date()}"
    )

    X_train = train_df[FEATURE_COLS]
    y_train = train_df['Y']
    X_val = val_df[FEATURE_COLS]
    y_val = val_df['Y']
    X_test = test_df[FEATURE_COLS]
    y_test = test_df['Y']

    # ---- Baseline: standalone Isolation Forest (original best single model) ----
    print("\n[Baseline] Isolation Forest (standalone, 3-feature subset as in original notebook)")
    iso_baseline = IsolationForest(n_estimators=500, contamination='auto', random_state=42)
    iso_baseline.fit(X_train[['VIX', 'USD_Over_100', 'XAU BGNL_cross']])
    iso_preds = iso_baseline.predict(X_test[['VIX', 'USD_Over_100', 'XAU BGNL_cross']])
    iso_preds = np.where(iso_preds == -1, 1, 0)
    print(classification_report(y_test, iso_preds, digits=4))
    baseline_f1 = f1_score(y_test, iso_preds)
    print(f"Baseline F1 (anomaly class): {baseline_f1:.4f}")

    # ---- Phase 1 ensemble ----
    print(f"\n[Phase 1] Training two-stage stacked ensemble (calibration={calibration!r})...")
    pipeline = AnomalyPipeline(calibration=calibration)
    pipeline.fit(X_train, y_train, X_val, y_val)

    print(f"\n[Calibration] Method chosen: {pipeline.calibration_method_}")
    print("[Calibration] OOF report (validation split only, selection criterion):")
    for method, scores in pipeline.calibration_report_.items():
        print(f"  {method:8s}  oof_brier={scores['oof_brier']:.4f}  oof_log_loss={scores['oof_log_loss']:.4f}")

    print("\n[Phase 1] Validation performance:")
    val_results = pipeline.evaluate(X_val, y_val, label="Validation")

    print("\n[Phase 1] Test performance:")
    test_results = pipeline.evaluate(X_test, y_test, label="Test")

    ensemble_f1 = test_results['f1']
    lift = (ensemble_f1 - baseline_f1) / baseline_f1 * 100
    print(f"\nF1 lift over baseline: {baseline_f1:.4f} -> {ensemble_f1:.4f} ({lift:+.1f}%)")

    # ---- Test Brier for both calibration methods (reporting only) ----
    print("\n--- (reporting only — not used for selection) ---")
    X_test_aug = pipeline._prepare(X_test)
    frozen = FrozenEstimator(pipeline._stacking)
    for method in ("sigmoid", "isotonic"):
        cal = CalibratedClassifierCV(frozen, method=method)
        X_val_aug = pipeline._prepare(X_val)
        cal.fit(X_val_aug, y_val)
        test_probas = cal.predict_proba(X_test_aug)[:, 1]
        test_brier = brier_score_loss(y_test, test_probas)
        test_f1 = f1_score(y_test, (test_probas >= 0.5).astype(int))
        print(f"  {method:8s}  test_brier={test_brier:.4f}  test_f1={test_f1:.4f}")

    # ---- Collect metrics for sidecar ----
    val_preds = val_results['predictions']
    test_preds = test_results['predictions']

    val_metrics = {
        "f1": float(f1_score(y_val, val_preds)),
        "precision": float(precision_score(y_val, val_preds, zero_division=0)),
        "recall": float(recall_score(y_val, val_preds, zero_division=0)),
    }
    test_metrics = {
        "f1": float(f1_score(y_test, test_preds)),
        "precision": float(precision_score(y_test, test_preds, zero_division=0)),
        "recall": float(recall_score(y_test, test_preds, zero_division=0)),
    }

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline.save(str(artifact_path))

    # ---- Write sidecar ----
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
            "train": {
                "start": str(train_df['Data'].min().date()),
                "end": str(train_df['Data'].max().date()),
            },
            "val": {
                "start": str(val_df['Data'].min().date()),
                "end": str(val_df['Data'].max().date()),
            },
            "test": {
                "start": str(test_df['Data'].min().date()),
                "end": str(test_df['Data'].max().date()),
            },
        },
        "calibration": {
            "chosen_method": pipeline.calibration_method_,
            "report": pipeline.calibration_report_,
        },
        "metrics": {
            "baseline_test_f1": float(baseline_f1),
            "ensemble_val": val_metrics,
            "ensemble_test": test_metrics,
        },
    }
    sidecar = artifact_path.with_suffix(".meta.json")
    sidecar.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Metadata written to {sidecar}")

    return pipeline


def main():
    parser = argparse.ArgumentParser(description="Train the Market Anomaly Detector (Phase 1)")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to EWS CSV")
    parser.add_argument("--out", type=Path, default=DEFAULT_ARTIFACT, help="Output path for saved pipeline")
    parser.add_argument(
        "--calibration",
        choices=["auto", "sigmoid", "isotonic"],
        default="auto",
        help="Calibration method (default: auto — selects by OOF Brier on val split)",
    )
    args = parser.parse_args()
    train(args.data, args.out, args.calibration)


if __name__ == "__main__":
    main()
