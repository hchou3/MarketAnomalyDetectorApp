"""Phase 2 Step 2.6 — SHAP explanations for the hgb detect head.

explain() runs shap.TreeExplainer on the raw clf margin (log-odds) and falls
back to shap.PermutationExplainer on calibrated probability when the tree
explainer is unavailable.

_rebuild_fold_detect_head() exactly mirrors evaluate_candidate's detect-head
training logic so that recomputed probabilities match the committed CSVs.

run_explain_cases() runs four case-study periods, verifies against committed
CSV predictions, and writes docs/reports/<date>-explanations.md.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

_INSTRUMENT_DISPLAY: dict[str, str] = {
    "vix": "VIX",
    "msci_usa": "S&P 500",
    "dollar_index": "Dollar Index",
    "usd_jpy": "USD/JPY",
    "gbp_usd": "GBP/USD",
    "gold": "Gold",
    "wti_front": "WTI crude",
    "us_10y_yield": "US 10Y",
    "us_2y_yield": "US 2Y",
    "us_3m_yield": "US 3M",
    "skew": "CBOE SKEW",
}

_CASE_STUDIES: list[tuple] = [
    ("2008 Sep–Dec", date(2008, 9, 1), date(2008, 12, 31), 2008, "dev"),
    ("2011 Aug–Oct", date(2011, 8, 1), date(2011, 10, 31), 2011, "dev"),
    ("2018 Q4", date(2018, 10, 1), date(2018, 12, 31), 2018, "confirm"),
    ("2020 Feb–Jun", date(2020, 2, 1), date(2020, 6, 30), 2020, "confirm"),
]


@dataclass
class TopFeature:
    rank: int
    col: str
    label: str
    contribution: float
    feature_value: float


@dataclass
class ExplainRow:
    week: date | None
    base_value: float
    contributions: dict[str, float]
    raw_margin: float
    calibrated_prob: float
    top_k_features: list[TopFeature]
    instrument_rollup: dict[str, float]
    explainer_used: str  # "tree" | "permutation"


def _normalize_binary_shap(
    sv,
    base_val,
) -> tuple[np.ndarray, float]:
    """Return (shap_values shape (n,f), base_value float) for class-1 direction.

    Handles four formats from shap.TreeExplainer:
    - list [arr_c0, arr_c1]: binary classifier returning per-class arrays
    - ndarray (n, f, 2): 3-D output with last axis = class
    - ndarray (n, f) with len-2 base_val [bv_c0, bv_c1]: per-class format
    - ndarray (n, f) with len-1 or scalar base_val: single-output decision_function space
    """

    def _bv(bv, *, want_idx1: bool) -> float:
        """Extract the float base value; want_idx1 picks class-1 from a 2-element sequence."""
        if hasattr(bv, "__len__"):
            if want_idx1 and len(bv) > 1:
                return float(bv[1])
            return float(bv[0])
        return float(bv)

    if isinstance(sv, list):
        arr = np.asarray(sv[1])
        return arr, _bv(base_val, want_idx1=True)
    sv = np.asarray(sv)
    if sv.ndim == 3:
        return sv[:, :, 1], _bv(base_val, want_idx1=True)
    # (n, f) — single-output or already class-1.
    # base_val is [bv_c1] (len==1, decision_function) or [bv_c0, bv_c1] (len==2).
    want_idx1 = hasattr(base_val, "__len__") and len(base_val) > 1
    return sv, _bv(base_val, want_idx1=want_idx1)


def explain(
    pipeline,
    platt,
    X_rows,
    feature_spec: list,
    weeks: list | None = None,
    top_k: int = 5,
    X_background: np.ndarray | None = None,
) -> list[ExplainRow]:
    """Compute SHAP explanations for each row in X_rows.

    Tries shap.TreeExplainer on the raw clf margin (log-odds contributions).
    Falls back to shap.PermutationExplainer on calibrated probability when
    TreeExplainer raises (e.g. model type not supported).

    Parameters
    ----------
    pipeline:
        Fitted sklearn Pipeline ending with a clf that has decision_function.
    platt:
        Fitted PlattCalibrator(use_logit=False) for this fold.
    X_rows:
        DataFrame or ndarray, shape (n_rows, n_features).  Columns must align
        with feature_spec in order.
    feature_spec:
        List of FeatureSpec objects describing each column of X_rows.
    weeks:
        Optional list of dates, one per row, stored in ExplainRow.week.
    top_k:
        Number of top features to store in top_k_features.
    X_background:
        Background dataset for PermutationExplainer fallback (≤ 100 rows used).
        If None and the fallback fires, the first min(100, n_rows) rows of
        X_rows are used.
    """
    import shap

    clf = pipeline.steps[-1][1]
    spec_cols = [s.col for s in feature_spec]

    if hasattr(X_rows, "values"):
        X_arr = X_rows.values.astype(float)
    else:
        X_arr = np.asarray(X_rows, dtype=float)

    raw_margins = clf.decision_function(X_arr)
    cal_probs = platt.transform(raw_margins)

    explainer_used = "tree"
    base_val_arr: np.ndarray

    try:
        tree_exp = shap.TreeExplainer(clf)
        sv = tree_exp.shap_values(X_arr, check_additivity=False)
        shap_values, bv_scalar = _normalize_binary_shap(sv, tree_exp.expected_value)
        base_val_arr = np.full(X_arr.shape[0], bv_scalar)
    except Exception:
        explainer_used = "permutation"
        if X_background is None:
            X_background = X_arr[: min(100, len(X_arr))]

        def _predict_fn(X: np.ndarray) -> np.ndarray:
            return platt.transform(clf.decision_function(np.asarray(X)))

        perm_exp = shap.PermutationExplainer(_predict_fn, X_background[:100], seed=0)
        sv_obj = perm_exp(X_arr)
        shap_values = np.asarray(sv_obj.values)
        bv_raw = sv_obj.base_values
        if hasattr(bv_raw, "__len__"):
            base_val_arr = np.asarray(bv_raw, dtype=float)
        else:
            base_val_arr = np.full(X_arr.shape[0], float(bv_raw))

    results: list[ExplainRow] = []
    for i in range(X_arr.shape[0]):
        sv_row = shap_values[i]  # (n_features,)
        contribs = {col: float(sv_row[j]) for j, col in enumerate(spec_cols)}

        sorted_feat_idx = sorted(
            enumerate(feature_spec),
            key=lambda x: -abs(contribs[x[1].col]),
        )
        top_k_list = [
            TopFeature(
                rank=rank,
                col=s.col,
                label=s.label,
                contribution=contribs[s.col],
                feature_value=float(X_arr[i, j]),
            )
            for rank, (j, s) in enumerate(sorted_feat_idx[:top_k], 1)
        ]

        rollup: dict[str, float] = {}
        for s in feature_spec:
            disp = _INSTRUMENT_DISPLAY.get(s.instrument_id, s.instrument_id)
            rollup[disp] = rollup.get(disp, 0.0) + contribs[s.col]

        results.append(
            ExplainRow(
                week=weeks[i] if weeks is not None else None,
                base_value=float(base_val_arr[i]),
                contributions=contribs,
                raw_margin=float(raw_margins[i]),
                calibrated_prob=float(cal_probs[i]),
                top_k_features=top_k_list,
                instrument_rollup=rollup,
                explainer_used=explainer_used,
            )
        )

    return results


def _rebuild_fold_detect_head(
    X_all: np.ndarray,
    detect_arr: np.ndarray,
    core_ok: np.ndarray,
    ku_mask: np.ndarray,
    fold,
):
    """Fit hgb detect head for one fold, exactly mirroring evaluate_candidate.

    Returns (pipeline, platt, threshold).  No data from the test window is used.
    """
    from .calibration import PlattCalibrator
    from .eval_harness import choose_threshold
    from .selection import FAR_BUDGET, make_hgb_pipeline

    core_train = core_ok[: fold.val_lo]
    core_val = core_ok[fold.val_lo : fold.test_lo]

    X_train = X_all[: fold.val_lo][core_train]
    y_train = detect_arr[: fold.val_lo][core_train]
    X_val = X_all[fold.val_lo : fold.test_lo][core_val]
    y_val = detect_arr[fold.val_lo : fold.test_lo][core_val]
    val_ku = ku_mask[fold.val_lo : fold.test_lo][core_val]

    pipe = make_hgb_pipeline()
    pipe.fit(X_train, y_train)

    val_raw = pipe.decision_function(X_val)
    platt = PlattCalibrator(use_logit=False).fit(val_raw, y_val)
    val_cal = platt.transform(val_raw)
    thr = choose_threshold(y_val, val_cal, FAR_BUDGET, excluded_mask=val_ku)

    return pipe, platt, thr


def _compute_mean_abs_contrib(
    X_all: np.ndarray,
    detect_arr: np.ndarray,
    core_ok: np.ndarray,
    ku_mask: np.ndarray,
    dev_folds: list,
) -> dict[str, float]:
    """Mean |SHAP contribution| per feature across all dev test weeks."""
    from .features.spec import SPEC_V1

    all_rows: list[ExplainRow] = []
    for fold in dev_folds:
        pipe, platt, _thr = _rebuild_fold_detect_head(X_all, detect_arr, core_ok, ku_mask, fold)
        core_test = core_ok[fold.test_lo : fold.test_hi]
        X_test = X_all[fold.test_lo : fold.test_hi][core_test]
        if len(X_test) == 0:
            continue
        core_train = core_ok[: fold.val_lo]
        X_train = X_all[: fold.val_lo][core_train]
        X_bg = X_train[: min(100, len(X_train))]
        rows = explain(pipe, platt, X_test, SPEC_V1, weeks=None, top_k=5, X_background=X_bg)
        all_rows.extend(rows)

    if not all_rows:
        return {s.col: 0.0 for s in SPEC_V1}

    result: dict[str, float] = {}
    for s in SPEC_V1:
        vals = [abs(r.contributions.get(s.col, 0.0)) for r in all_rows]
        result[s.col] = float(np.mean(vals)) if vals else 0.0
    return result


def _write_explain_report(
    out_path: Path,
    case_results: list[dict],
    mean_abs_contrib: dict[str, float],
    label_set: str,
    today: str,
) -> None:
    from .features.spec import SPEC_V1

    lines = [
        f"# SHAP Explanations — HGB Detect Head ({label_set})",
        "",
        f"Generated: {today}  ",
        (
            f"Label set: `{label_set}` · Model: hgb · "
            "SHAP: tree path (log-odds) with permutation fallback  "
        ),
        "",
        "Contributions in log-odds space (tree path): positive = toward stress,  ",
        "negative = toward calm.  base\\_value + Σ contributions = raw margin.  ",
        "",
        "---",
        "",
        "## Case studies",
        "",
    ]

    for case in case_results:
        rows: list[ExplainRow] = case["rows"]
        name: str = case["name"]
        lines += [f"### {name}", ""]

        if not rows:
            lines += ["*(no data)*", ""]
            continue

        explainer = rows[0].explainer_used
        lines += [
            f"Fold year: {case['fold_year']} · Weeks: {len(rows)} · Explainer: {explainer}  ",
            "",
            "| Week | Prob | Margin | Alert? | Top-5 features (contribution) |",
            "|------|------|--------|--------|-------------------------------|",
        ]

        for r in rows:
            top5 = "; ".join(f"{f.col} ({f.contribution:+.3f})" for f in r.top_k_features)
            lines.append(
                f"| {r.week} | {r.calibrated_prob:.3f} | {r.raw_margin:+.3f} | — | {top5} |"
            )

        lines += ["", "#### Instrument rollup (mean contribution across case weeks)", ""]
        lines += [
            "| Instrument | Mean SHAP (log-odds) |",
            "|-----------|---------------------|",
        ]
        rollup_total: dict[str, float] = {}
        for r in rows:
            for k, v in r.instrument_rollup.items():
                rollup_total[k] = rollup_total.get(k, 0.0) + v
        rollup_avg = {k: v / len(rows) for k, v in rollup_total.items()}
        for k, v in sorted(rollup_avg.items(), key=lambda x: -abs(x[1])):
            lines.append(f"| {k} | {v:+.4f} |")
        lines.append("")

    lines += [
        "---",
        "",
        "## Dev-period mean |SHAP| (feature importance, 2008–2017)",
        "",
        "Mean absolute SHAP contribution across all dev fold test weeks.  ",
        "",
        "| Rank | Feature | Label | Mean |SHAP| |",
        "|------|---------|-------|------------|",
    ]

    sorted_feats = sorted(SPEC_V1, key=lambda s: -mean_abs_contrib.get(s.col, 0.0))
    for rank, s in enumerate(sorted_feats, 1):
        v = mean_abs_contrib.get(s.col, 0.0)
        lines.append(f"| {rank} | {s.col} | {s.label} | {v:.5f} |")

    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_explain_cases(
    session,
    out_dir,
    label_set: str = "ews_v1",
    csv_path=None,
) -> int:
    """Run SHAP case studies on 4 stress episodes; write docs/reports/<date>-explanations.md.

    For each case study:
    1. Rebuilds the fold model (same logic as evaluate_candidate detect head).
    2. Verifies recomputed calibrated probabilities match committed CSV within 1e-9.
    3. Runs SHAP on the case-study weeks.

    Then computes mean |contribution| across all dev test weeks and writes a report.
    """
    import pandas as pd

    from .eval_harness import KNOWN_UNLABELED, make_folds_v2
    from .features.build import build_from_db
    from .features.spec import SPEC_V1
    from .labels import load_labels
    from .selection import _core_feature_mask

    labels_df = load_labels(session, label_set=label_set)
    if labels_df.empty:
        print(f"ERROR: no labels for {label_set!r}", file=sys.stderr)
        return 1

    weeks = list(labels_df["week"])
    detect_arr = labels_df["value"].values.astype(int)
    X_df = build_from_db(session, SPEC_V1, weeks, as_of=None)
    X_all = X_df.values.astype(float)
    core_ok = _core_feature_mask(X_df, X_all)

    ku_mask = np.zeros(len(weeks), dtype=bool)
    for ks, ke, _ in KNOWN_UNLABELED:
        for i, w in enumerate(weeks):
            if ks <= w <= ke:
                ku_mask[i] = True

    dev_folds = make_folds_v2(weeks, detect_arr, first_test_year=2008, last_test_year=2017)
    confirm_folds = make_folds_v2(weeks, detect_arr, first_test_year=2018, last_test_year=2021)

    # Load probability verification maps from committed CSVs
    repo_root = Path(__file__).parent.parent.parent.parent
    artifacts_dir = repo_root / "artifacts"
    verify_map: dict[date, float] = {}

    confirm_csv_p = Path(csv_path) if csv_path else (artifacts_dir / "confirm_predictions.csv")
    for csv_p in [artifacts_dir / "dev_predictions.csv", confirm_csv_p]:
        if csv_p.exists():
            df_v = pd.read_csv(str(csv_p))
            if "hgb_prob" in df_v.columns and "week" in df_v.columns:
                for _, row in df_v.iterrows():
                    w = date.fromisoformat(str(row["week"])[:10])
                    verify_map[w] = float(row["hgb_prob"])

    case_results: list[dict] = []

    for case_name, start_d, end_d, fold_year, split in _CASE_STUDIES:
        folds = dev_folds if split == "dev" else confirm_folds
        fold = next((f for f in folds if f.test_year == fold_year), None)
        if fold is None:
            print(f"WARNING: no fold for year {fold_year}, skipping {case_name}")
            continue

        pipe, platt, _thr = _rebuild_fold_detect_head(X_all, detect_arr, core_ok, ku_mask, fold)

        test_weeks_all = weeks[fold.test_lo : fold.test_hi]
        core_test = core_ok[fold.test_lo : fold.test_hi]
        X_test = X_all[fold.test_lo : fold.test_hi][core_test]
        test_weeks = [w for w, ok in zip(test_weeks_all, core_test) if ok]

        if len(X_test) == 0:
            print(f"WARNING: no core-ok rows for fold {fold_year}, skipping {case_name}")
            continue

        raw_all = pipe.decision_function(X_test)
        cal_all = platt.transform(raw_all)

        for w, prob in zip(test_weeks, cal_all):
            if w in verify_map:
                diff = abs(prob - verify_map[w])
                if diff > 1e-9:
                    print(
                        f"ERROR: {case_name}: week {w} mismatch — "
                        f"recomputed={prob:.12f} csv={verify_map[w]:.12f} diff={diff:.2e}",
                        file=sys.stderr,
                    )
                    return 1

        case_mask = np.array([start_d <= w <= end_d for w in test_weeks])
        X_case = X_test[case_mask]
        case_weeks = [w for w, ok in zip(test_weeks, case_mask) if ok]

        if len(case_weeks) == 0:
            print(f"WARNING: no weeks in {case_name} window, skipping")
            continue

        core_train = core_ok[: fold.val_lo]
        X_train = X_all[: fold.val_lo][core_train]
        X_background = X_train[: min(100, len(X_train))]

        rows = explain(
            pipe, platt, X_case, SPEC_V1, weeks=case_weeks, top_k=5, X_background=X_background
        )
        case_results.append({"name": case_name, "rows": rows, "fold_year": fold_year})
        print(
            f"  {case_name}: {len(rows)} week(s), "
            f"explainer={rows[0].explainer_used if rows else 'n/a'}"
        )

    if not case_results:
        print("ERROR: no case studies completed", file=sys.stderr)
        return 1

    print("Computing mean |SHAP| across all dev test weeks (rebuilds 10 fold models)…")
    mean_abs_contrib = _compute_mean_abs_contrib(X_all, detect_arr, core_ok, ku_mask, dev_folds)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    out_path = out_dir / f"{today}-explanations.md"
    _write_explain_report(out_path, case_results, mean_abs_contrib, label_set, today)
    print(f"explanations: {len(case_results)} case(s) → {out_path}")
    return 0
