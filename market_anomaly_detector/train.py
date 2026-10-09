"""
Train the two-stage anomaly pipeline and save all artifacts.

Usage:
    python -m market_anomaly_detector.train --data "data/FinancialMarketData.xlsx - EWS.csv"
    python -m market_anomaly_detector.train
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, f1_score

from .data import engineer_features, load_data, FEATURE_COLS
from .pipeline import AnomalyPipeline
from .splits import split_timeseries

DEFAULT_DATA = Path(__file__).parent.parent / "data" / "FinancialMarketData.xlsx - EWS.csv"
DEFAULT_ARTIFACT = Path(__file__).parent.parent / "artifacts" / "pipeline.joblib"


def train(data_path: Path = DEFAULT_DATA, artifact_path: Path = DEFAULT_ARTIFACT) -> AnomalyPipeline:
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
    print("\n[Phase 1] Training two-stage stacked ensemble...")
    pipeline = AnomalyPipeline()
    pipeline.fit(X_train, y_train, X_val, y_val)

    print("\n[Phase 1] Validation performance:")
    val_results = pipeline.evaluate(X_val, y_val, label="Validation")

    print("\n[Phase 1] Test performance:")
    test_results = pipeline.evaluate(X_test, y_test, label="Test")

    ensemble_f1 = test_results['f1']
    lift = (ensemble_f1 - baseline_f1) / baseline_f1 * 100
    print(f"\nF1 lift over baseline: {baseline_f1:.4f} -> {ensemble_f1:.4f} ({lift:+.1f}%)")

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline.save(str(artifact_path))
    return pipeline


def main():
    parser = argparse.ArgumentParser(description="Train the Market Anomaly Detector (Phase 1)")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to EWS CSV")
    parser.add_argument("--out", type=Path, default=DEFAULT_ARTIFACT, help="Output path for saved pipeline")
    args = parser.parse_args()
    train(args.data, args.out)


if __name__ == "__main__":
    main()
