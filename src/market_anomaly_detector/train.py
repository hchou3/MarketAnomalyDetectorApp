"""
Train the two-stage anomaly pipeline and save all artifacts.

Usage:
    mad-train
    mad-train --data data/raw/ews_weekly_2000_2021.csv
"""
import argparse
import logging
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, f1_score

from .data import engineer_features, load_data, FEATURE_COLS
from .paths import ARTIFACT_DIR, DATA_PATH
from .pipeline import AnomalyPipeline
from .splits import split_timeseries

log = logging.getLogger(__name__)


def train(data_path: Path = DATA_PATH, artifact_path: Path = ARTIFACT_DIR / "pipeline.joblib") -> AnomalyPipeline:
    log.info("Loading data from %s", data_path)
    raw = load_data(str(data_path))
    featured = engineer_features(raw)
    log.info("After feature engineering: %d rows, %d features", len(featured), len(FEATURE_COLS))

    train_df, val_df, test_df = split_timeseries(featured, train_years=9, val_years=2, total_years=12)
    log.info(
        "Split sizes -- train: %d, val: %d, test: %d\n"
        "  Train: %s to %s\n"
        "  Val:   %s to %s\n"
        "  Test:  %s to %s",
        len(train_df), len(val_df), len(test_df),
        train_df["Data"].min().date(), train_df["Data"].max().date(),
        val_df["Data"].min().date(), val_df["Data"].max().date(),
        test_df["Data"].min().date(), test_df["Data"].max().date(),
    )

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["Y"]
    X_val = val_df[FEATURE_COLS]
    y_val = val_df["Y"]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df["Y"]

    log.info("[Baseline] Isolation Forest (standalone, 3-feature subset as in original notebook)")
    iso_baseline = IsolationForest(n_estimators=500, contamination="auto", random_state=42)
    iso_baseline.fit(X_train[["VIX", "USD_Over_100", "XAU BGNL_cross"]])
    iso_preds = iso_baseline.predict(X_test[["VIX", "USD_Over_100", "XAU BGNL_cross"]])
    iso_preds = np.where(iso_preds == -1, 1, 0)
    log.info("\n%s", classification_report(y_test, iso_preds, digits=4))
    baseline_f1 = f1_score(y_test, iso_preds)
    log.info("Baseline F1 (anomaly class): %.4f", baseline_f1)

    log.info("[Phase 1] Training two-stage stacked ensemble...")
    pipeline = AnomalyPipeline()
    pipeline.fit(X_train, y_train, X_val, y_val)

    log.info("[Phase 1] Validation performance:")
    val_results = pipeline.evaluate(X_val, y_val, label="Validation")

    log.info("[Phase 1] Test performance:")
    test_results = pipeline.evaluate(X_test, y_test, label="Test")

    ensemble_f1 = test_results["f1"]
    lift = (ensemble_f1 - baseline_f1) / baseline_f1 * 100
    log.info("F1 lift over baseline: %.4f -> %.4f (%+.1f%%)", baseline_f1, ensemble_f1, lift)

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline.save(str(artifact_path))
    return pipeline


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Train the Market Anomaly Detector (Phase 1)")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Path to EWS CSV")
    parser.add_argument(
        "--out", type=Path, default=ARTIFACT_DIR / "pipeline.joblib", help="Output path for saved pipeline"
    )
    args = parser.parse_args()
    train(args.data, args.out)


if __name__ == "__main__":
    main()
