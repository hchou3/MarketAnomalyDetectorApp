"""
Walk-forward (expanding-window) backtest over the full 2000-2021 history.

The single 9y/2y/1y split used by ``train.py`` scores only one test year
(52 weeks, May 2011 - May 2012) and leaves 2012-2021 unused. This module
re-runs the *exact same training procedure* once per calendar year, always
training only on the past, and pools every out-of-sample prediction so the
ensemble-vs-baseline comparison rests on ~14 years of unseen data instead of one.

Fold construction (one fold per calendar test year Y):

    train = every week before the calibration window
    val   = the ``val_years`` calendar years immediately before Y, extended
            backwards one year at a time until it holds at least
            ``min_val_positives`` anomaly weeks AND as many normal weeks
            (calibration cannot be fitted on a window with ~no anomalies,
            and several 2-year windows in this data have only 0-6)
    test  = calendar year Y

Only past labels decide the window extension, so no future information is used.
Train < val < test holds strictly for every fold (see tests/test_backtest.py).

Because anomalies arrive in bursts (several calendar years contain none), the
headline metrics are *pooled* over all out-of-sample weeks. Per-year rows report
counts (caught / missed / false alarms) rather than F1, which is undefined or
meaningless in a year with no anomalies.

Usage:
    python -m market_anomaly_detector.backtest
    python -m market_anomaly_detector.backtest --calibration sigmoid --out-dir artifacts/backtest_sigmoid
"""
from __future__ import annotations

import argparse
import json
import platform
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .data import FEATURE_COLS, engineer_features, load_data
from .pipeline import AnomalyPipeline

DEFAULT_DATA = Path(__file__).parent.parent / "data" / "FinancialMarketData.xlsx - EWS.csv"
DEFAULT_OUT_DIR = Path(__file__).parent.parent / "artifacts" / "backtest"

# Same 3-feature standalone Isolation Forest that train.py uses as the baseline
BASELINE_FEATURES = ['VIX', 'USD_Over_100', 'XAU BGNL_cross']


# ----------------------------------------------------------------------
# Fold construction
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class Fold:
    """Row-position boundaries into a date-sorted featured frame.

    train = [0, val_lo), val = [val_lo, test_lo), test = [test_lo, test_hi)
    """
    test_year: int
    val_lo: int
    test_lo: int
    test_hi: int
    val_start_year: int
    val_positives: int
    val_negatives: int

    @property
    def val_years_used(self) -> int:
        return self.test_year - self.val_start_year


def make_folds(
    featured: pd.DataFrame,
    first_test_year: int = 2008,
    last_test_year: int | None = None,
    val_years: int = 2,
    min_val_positives: int = 8,
    min_train_rows: int = 104,
) -> list[Fold]:
    """Build one expanding-window fold per calendar test year."""
    if not featured['Data'].is_monotonic_increasing:
        raise ValueError("featured must be sorted by 'Data'.")

    years = featured['Data'].dt.year.to_numpy()
    y = featured['Y'].to_numpy()
    first_year_in_data = int(years.min())
    if last_test_year is None:
        last_test_year = int(years.max())

    folds: list[Fold] = []
    for test_year in range(first_test_year, last_test_year + 1):
        test_pos = np.flatnonzero(years == test_year)
        if test_pos.size == 0:
            continue

        val_start_year = test_year - val_years
        while True:
            in_val = (years >= val_start_year) & (years < test_year)
            pos = int(y[in_val].sum())
            neg = int(in_val.sum() - pos)
            if pos >= min_val_positives and neg >= min_val_positives:
                break
            val_start_year -= 1
            if val_start_year <= first_year_in_data:
                raise ValueError(
                    f"Test year {test_year}: cannot find a calibration window with "
                    f">= {min_val_positives} anomalies while leaving training data."
                )

        val_lo = int(np.flatnonzero(years >= val_start_year)[0])
        test_lo, test_hi = int(test_pos[0]), int(test_pos[-1]) + 1

        train_y = y[:val_lo]
        if val_lo < min_train_rows or train_y.sum() < 2 or (train_y == 0).sum() < 2:
            raise ValueError(
                f"Test year {test_year}: training window too small or single-class "
                f"({val_lo} rows, {int(train_y.sum())} anomalies). Raise first_test_year."
            )

        folds.append(Fold(
            test_year=test_year,
            val_lo=val_lo,
            test_lo=test_lo,
            test_hi=test_hi,
            val_start_year=val_start_year,
            val_positives=pos,
            val_negatives=neg,
        ))
    return folds


# ----------------------------------------------------------------------
# Running one fold
# ----------------------------------------------------------------------

def run_fold(featured: pd.DataFrame, fold: Fold, calibration: str = "auto") -> tuple[pd.DataFrame, dict]:
    """Fit the ensemble and the baseline on the fold's past; score its test year."""
    train = featured.iloc[:fold.val_lo]
    val = featured.iloc[fold.val_lo:fold.test_lo]
    test = featured.iloc[fold.test_lo:fold.test_hi]

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        # Small calibration windows trigger sklearn's "least populated class" notice.
        warnings.filterwarnings("ignore", message=".*least populated class.*")
        pipe = AnomalyPipeline(calibration=calibration)
        pipe.fit(train[FEATURE_COLS], train['Y'], val[FEATURE_COLS], val['Y'])
    fit_seconds = time.perf_counter() - t0

    ens_proba = pipe.predict_proba(test[FEATURE_COLS])[:, 1]

    iso = IsolationForest(n_estimators=500, contamination='auto', random_state=42)
    iso.fit(train[BASELINE_FEATURES])
    base_pred = (iso.predict(test[BASELINE_FEATURES]) == -1).astype(int)
    base_score = -iso.decision_function(test[BASELINE_FEATURES])  # higher = more anomalous

    preds = pd.DataFrame({
        "Data": test['Data'].to_numpy(),
        "test_year": fold.test_year,
        "Y": test['Y'].to_numpy().astype(int),
        "ens_proba": ens_proba,
        "ens_pred": (ens_proba >= 0.5).astype(int),
        "base_score": base_score,
        "base_pred": base_pred,
    })
    info = {
        **asdict(fold),
        "train_start": str(train['Data'].iloc[0].date()),
        "train_end": str(train['Data'].iloc[-1].date()),
        "val_start": str(val['Data'].iloc[0].date()),
        "val_end": str(val['Data'].iloc[-1].date()),
        "test_start": str(test['Data'].iloc[0].date()),
        "test_end": str(test['Data'].iloc[-1].date()),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "calibration_method": pipe.calibration_method_,
        "calibration_report": pipe.calibration_report_,
        "fit_seconds": round(fit_seconds, 2),
    }
    return preds, info


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def confusion_metrics(y_true, y_pred) -> dict:
    """Counts plus precision / recall / F1 / false-alarm rate, safe for empty classes."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    far = fp / (fp + tn) if fp + tn else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1, "false_alarm_rate": far,
    }


def year_block_bootstrap(
    preds: pd.DataFrame,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """
    Resample whole test years with replacement and recompute pooled F1 for the
    ensemble and the baseline. Resampling years (not weeks) respects the strong
    week-to-week autocorrelation of crisis periods.
    """
    rng = np.random.default_rng(seed)
    groups = {yr: g for yr, g in preds.groupby("test_year")}
    yrs = np.array(sorted(groups))

    # Pre-compute per-year confusion counts so each replicate is just a sum.
    def counts(g, col):
        m = confusion_metrics(g['Y'], g[col])
        return np.array([m['tp'], m['fp'], m['fn']])

    ens_c = {yr: counts(groups[yr], 'ens_pred') for yr in yrs}
    base_c = {yr: counts(groups[yr], 'base_pred') for yr in yrs}

    def f1_from(c):
        tp, fp, fn = c
        return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0

    ens_f1, base_f1 = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.choice(yrs, size=len(yrs), replace=True)
        ens_f1[b] = f1_from(sum(ens_c[s] for s in sample))
        base_f1[b] = f1_from(sum(base_c[s] for s in sample))
    diff = ens_f1 - base_f1

    def ci(a):
        lo, hi = np.percentile(a, [2.5, 97.5])
        return [float(lo), float(hi)]

    return {
        "n_boot": n_boot,
        "unit": "calendar test year",
        "ensemble_f1_ci95": ci(ens_f1),
        "baseline_f1_ci95": ci(base_f1),
        "f1_diff_ci95": ci(diff),
        "p_ensemble_better": float((diff > 0).mean()),
    }


def summarise(preds: pd.DataFrame, fold_infos: list[dict], n_boot: int = 2000) -> tuple[dict, pd.DataFrame]:
    """Pooled out-of-sample metrics + per-year table."""
    y = preds['Y'].to_numpy()
    ens = confusion_metrics(y, preds['ens_pred'])
    base = confusion_metrics(y, preds['base_pred'])
    always = confusion_metrics(y, np.ones_like(y))

    two_classes = len(np.unique(y)) == 2
    threshold_free = {
        "ensemble_pr_auc": float(average_precision_score(y, preds['ens_proba'])) if two_classes else None,
        "baseline_pr_auc": float(average_precision_score(y, preds['base_score'])) if two_classes else None,
        "ensemble_roc_auc": float(roc_auc_score(y, preds['ens_proba'])) if two_classes else None,
        "baseline_roc_auc": float(roc_auc_score(y, preds['base_score'])) if two_classes else None,
        "prevalence": float(y.mean()),
    }

    info_by_year = {f['test_year']: f for f in fold_infos}
    rows = []
    for yr, g in preds.groupby("test_year"):
        e = confusion_metrics(g['Y'], g['ens_pred'])
        b = confusion_metrics(g['Y'], g['base_pred'])
        inf = info_by_year[yr]
        rows.append({
            "test_year": yr,
            "weeks": len(g),
            "anomalies": int(g['Y'].sum()),
            "ens_caught": e['tp'], "ens_missed": e['fn'], "ens_false_alarms": e['fp'],
            "base_caught": b['tp'], "base_missed": b['fn'], "base_false_alarms": b['fp'],
            "ens_mean_proba": round(float(g['ens_proba'].mean()), 4),
            "calibration": inf['calibration_method'],
            "val_window": f"{inf['val_start_year']}-{yr - 1}",
            "val_anomalies": inf['val_positives'],
            "train_weeks": inf['n_train'],
        })
    per_year = pd.DataFrame(rows)

    summary = {
        "n_folds": len(fold_infos),
        "test_period": [str(preds['Data'].min().date()), str(preds['Data'].max().date())],
        "n_test_weeks": int(len(preds)),
        "n_test_anomalies": int(y.sum()),
        "pooled": {
            "ensemble": ens,
            "baseline_isolation_forest": base,
            "always_flag_anomaly": always,
            "ensemble_brier": float(brier_score_loss(y, preds['ens_proba'])),
        },
        "threshold_free": threshold_free,
        "per_year_f1": {
            "note": "Mean/std over years that contain at least one anomaly; F1 is undefined otherwise.",
            "ensemble_mean": None, "ensemble_std": None,
            "baseline_mean": None, "baseline_std": None,
            "years_ensemble_better": None, "years_baseline_better": None, "years_tied": None,
            "n_years": None,
        },
        "bootstrap": year_block_bootstrap(preds, n_boot=n_boot) if n_boot else None,
        "calibration_methods_chosen": per_year['calibration'].value_counts().to_dict(),
    }

    yr_f1 = []
    for yr, g in preds.groupby("test_year"):
        if g['Y'].sum() == 0:
            continue
        yr_f1.append((confusion_metrics(g['Y'], g['ens_pred'])['f1'],
                      confusion_metrics(g['Y'], g['base_pred'])['f1']))
    if yr_f1:
        a = np.array(yr_f1)
        d = a[:, 0] - a[:, 1]
        summary["per_year_f1"].update({
            "ensemble_mean": float(a[:, 0].mean()), "ensemble_std": float(a[:, 0].std(ddof=1)) if len(a) > 1 else 0.0,
            "baseline_mean": float(a[:, 1].mean()), "baseline_std": float(a[:, 1].std(ddof=1)) if len(a) > 1 else 0.0,
            "years_ensemble_better": int((d > 1e-12).sum()),
            "years_baseline_better": int((d < -1e-12).sum()),
            "years_tied": int((np.abs(d) <= 1e-12).sum()),
            "n_years": len(a),
        })
    return summary, per_year


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------

def run_backtest(
    featured: pd.DataFrame,
    first_test_year: int = 2008,
    last_test_year: int | None = None,
    val_years: int = 2,
    min_val_positives: int = 8,
    calibration: str = "auto",
    n_boot: int = 2000,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Returns (per-week predictions, per-year table, summary dict)."""
    folds = make_folds(featured, first_test_year, last_test_year, val_years, min_val_positives)
    all_preds, infos = [], []
    for fold in folds:
        preds, info = run_fold(featured, fold, calibration)
        all_preds.append(preds)
        infos.append(info)
        if verbose:
            print(
                f"  {fold.test_year}: train {info['train_start']}..{info['train_end']} "
                f"({info['n_train']}w) | val {info['val_start']}..{info['val_end']} "
                f"({info['val_positives']} anomalies) | test {info['n_test']}w, "
                f"{int(preds['Y'].sum())} anomalies | cal={info['calibration_method']} "
                f"| {info['fit_seconds']}s"
            )
    preds = pd.concat(all_preds, ignore_index=True)
    summary, per_year = summarise(preds, infos, n_boot=n_boot)
    summary["config"] = {
        "first_test_year": first_test_year,
        "last_test_year": last_test_year,
        "val_years": val_years,
        "min_val_positives": min_val_positives,
        "calibration": calibration,
        "decision_threshold": 0.5,
        "baseline_features": BASELINE_FEATURES,
    }
    summary["folds"] = infos
    return preds, per_year, summary


def _print_report(summary: dict, per_year: pd.DataFrame) -> None:
    p = summary["pooled"]
    e, b, a = p["ensemble"], p["baseline_isolation_forest"], p["always_flag_anomaly"]
    tf = summary["threshold_free"]
    print("\n=== Per test year (counts of weeks) ===")
    cols = ["test_year", "weeks", "anomalies", "ens_caught", "ens_false_alarms",
            "base_caught", "base_false_alarms", "calibration", "val_window"]
    print(per_year[cols].to_string(index=False))

    print(f"\n=== Pooled out-of-sample: {summary['test_period'][0]} .. {summary['test_period'][1]} "
          f"({summary['n_test_weeks']} weeks, {summary['n_test_anomalies']} anomalies, "
          f"prevalence {tf['prevalence']:.1%}) ===")
    print(f"{'':28s}{'F1':>7s}{'Prec':>7s}{'Recall':>8s}{'FalseAlarmRate':>16s}{'caught':>8s}{'false alarms':>14s}")
    for name, m in [("Ensemble", e), ("Baseline Isolation Forest", b), ("Always flag (floor)", a)]:
        print(f"{name:28s}{m['f1']:7.3f}{m['precision']:7.3f}{m['recall']:8.3f}"
              f"{m['false_alarm_rate']:16.3f}{m['tp']:8d}{m['fp']:14d}")
    print(f"\nThreshold-free  PR-AUC: ensemble {tf['ensemble_pr_auc']:.3f} vs baseline {tf['baseline_pr_auc']:.3f}"
          f"  |  ROC-AUC: ensemble {tf['ensemble_roc_auc']:.3f} vs baseline {tf['baseline_roc_auc']:.3f}")
    print(f"Ensemble Brier score: {p['ensemble_brier']:.4f}")

    py = summary["per_year_f1"]
    if py["n_years"]:
        print(f"\nPer-year F1 over {py['n_years']} years with anomalies: ensemble {py['ensemble_mean']:.3f} "
              f"± {py['ensemble_std']:.3f}, baseline {py['baseline_mean']:.3f} ± {py['baseline_std']:.3f} "
              f"(ensemble better in {py['years_ensemble_better']}, baseline in {py['years_baseline_better']}, "
              f"tied {py['years_tied']})")
    bs = summary["bootstrap"]
    if bs:
        lo, hi = bs["f1_diff_ci95"]
        print(f"Year-block bootstrap ({bs['n_boot']} reps): pooled F1 difference (ensemble - baseline) "
              f"95% CI [{lo:+.3f}, {hi:+.3f}], P(ensemble better) = {bs['p_ensemble_better']:.2f}")
    print(f"Calibration methods chosen per fold: {summary['calibration_methods_chosen']}")


def main():
    parser = argparse.ArgumentParser(description="Walk-forward backtest of the Market Anomaly Detector")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--first-test-year", type=int, default=2008)
    parser.add_argument("--last-test-year", type=int, default=None)
    parser.add_argument("--val-years", type=int, default=2)
    parser.add_argument("--min-val-positives", type=int, default=8)
    parser.add_argument("--calibration", choices=["auto", "sigmoid", "isotonic"], default="auto")
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()

    import joblib, shap, sklearn  # noqa: E401  (versions for the record)

    featured = engineer_features(load_data(str(args.data)))
    print(f"Walk-forward backtest: {len(featured)} weeks, test years {args.first_test_year}.."
          f"{args.last_test_year or int(featured['Data'].dt.year.max())}, calibration={args.calibration!r}")
    preds, per_year, summary = run_backtest(
        featured,
        first_test_year=args.first_test_year,
        last_test_year=args.last_test_year,
        val_years=args.val_years,
        min_val_positives=args.min_val_positives,
        calibration=args.calibration,
        n_boot=args.n_boot,
    )
    summary["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    summary["versions"] = {
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__, "shap": shap.__version__, "joblib": joblib.__version__,
    }

    _print_report(summary, per_year)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    preds.to_csv(args.out_dir / "predictions.csv", index=False)
    per_year.to_csv(args.out_dir / "per_year.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote predictions.csv, per_year.csv, summary.json to {args.out_dir}")


if __name__ == "__main__":
    main()
