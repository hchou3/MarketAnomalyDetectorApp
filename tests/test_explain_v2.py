"""Tests for explain_v2.py — SHAP explanations for the hgb detect head (Step 2.6).

Tests cover:
- SHAP additivity (base + sum(contributions) == raw_margin) on tree path
- Calibrated probability consistency with PlattCalibrator
- Instrument rollup sums to total contributions
- Top-k ordering by |contribution| (descending)
- Top-k count equals min(top_k, n_features)
- SPEC_V1 labels coverage (all non-empty)
- Permutation fallback when TreeExplainer raises
- Fold rebuild determinism (bitwise-identical probabilities)
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

# ── Shared helpers ────────────────────────────────────────────────────────────


def _make_hgb_pipeline_and_platt(n: int = 200, n_features: int = 10, seed: int = 0):
    """Tiny HGB pipeline + fitted PlattCalibrator on synthetic data."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import Pipeline

    from market_anomaly_detector.calibration import PlattCalibrator

    rng = np.random.default_rng(seed)
    X_train = rng.standard_normal((n, n_features))
    y_train = (rng.random(n) < 0.25).astype(int)
    y_train[:10] = 1  # guarantee positives in training set

    pipe = Pipeline(
        [("clf", HistGradientBoostingClassifier(max_depth=3, max_iter=50, random_state=42))]
    )
    pipe.fit(X_train, y_train)

    val_raw = pipe.decision_function(X_train[:50])
    platt = PlattCalibrator(use_logit=False).fit(val_raw, y_train[:50])
    return pipe, platt, X_train


def _make_feature_spec(n_features: int = 10):
    """Synthetic FeatureSpec list with n_features entries across several instruments."""
    from market_anomaly_detector.features.spec import FeatureSpec

    instruments = [
        "vix",
        "msci_usa",
        "dollar_index",
        "usd_jpy",
        "gbp_usd",
        "gold",
        "wti_front",
        "us_10y_yield",
        "us_2y_yield",
        "us_3m_yield",
    ]
    return [
        FeatureSpec(
            col=f"feat_{i}",
            label=f"Synthetic feature {i}",
            instrument_id=instruments[i % len(instruments)],
            units="index_points",
            transform="log_return",
            window=1,
        )
        for i in range(n_features)
    ]


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_additivity_tree_path():
    """base_value + sum(contributions) == raw_margin within 1e-6 (tree path)."""
    from market_anomaly_detector.explain_v2 import explain

    pipe, platt, X_train = _make_hgb_pipeline_and_platt(n=200, n_features=10)
    spec = _make_feature_spec(n_features=10)
    X_test = X_train[100:110]

    rows = explain(pipe, platt, X_test, spec)
    assert rows, "explain() returned an empty list"

    tree_rows = [r for r in rows if r.explainer_used == "tree"]
    if not tree_rows:
        pytest.skip("TreeExplainer not available — additivity requires tree path")

    for r in tree_rows:
        total = r.base_value + sum(r.contributions.values())
        assert abs(total - r.raw_margin) < 1e-6, (
            f"Additivity violated: base={r.base_value:.8f}"
            f" + Σcontribs={sum(r.contributions.values()):.8f}"
            f" = {total:.8f} ≠ raw_margin={r.raw_margin:.8f}"
        )


def test_calibration_consistent():
    """calibrated_prob == platt.transform(raw_margin) for every row."""
    from market_anomaly_detector.explain_v2 import explain

    pipe, platt, X_train = _make_hgb_pipeline_and_platt(n=200, n_features=10)
    spec = _make_feature_spec(n_features=10)
    rows = explain(pipe, platt, X_train[100:110], spec)

    for r in rows:
        expected = float(platt.transform(np.array([r.raw_margin]))[0])
        assert abs(r.calibrated_prob - expected) < 1e-9, (
            f"Calibration mismatch: stored={r.calibrated_prob:.12f} recomputed={expected:.12f}"
        )


def test_instrument_rollup_sums_to_total():
    """sum(rollup.values()) == sum(contributions.values()) within 1e-9."""
    from market_anomaly_detector.explain_v2 import explain

    pipe, platt, X_train = _make_hgb_pipeline_and_platt(n=200, n_features=10)
    spec = _make_feature_spec(n_features=10)
    rows = explain(pipe, platt, X_train[100:110], spec)

    for r in rows:
        rollup_sum = sum(r.instrument_rollup.values())
        contribs_sum = sum(r.contributions.values())
        assert abs(rollup_sum - contribs_sum) < 1e-9, (
            f"rollup_sum={rollup_sum:.12f} != contribs_sum={contribs_sum:.12f}"
        )


def test_top_k_ordered_by_abs_contribution():
    """top_k_features is sorted descending by |contribution|."""
    from market_anomaly_detector.explain_v2 import explain

    pipe, platt, X_train = _make_hgb_pipeline_and_platt(n=200, n_features=10)
    spec = _make_feature_spec(n_features=10)
    rows = explain(pipe, platt, X_train[100:110], spec, top_k=5)

    for r in rows:
        abs_contribs = [abs(f.contribution) for f in r.top_k_features]
        assert abs_contribs == sorted(abs_contribs, reverse=True), (
            f"top_k_features not sorted by |contribution|: {abs_contribs}"
        )


def test_top_k_count():
    """len(top_k_features) == min(top_k, n_features) for all top_k values."""
    from market_anomaly_detector.explain_v2 import explain

    pipe, platt, X_train = _make_hgb_pipeline_and_platt(n=200, n_features=10)
    spec = _make_feature_spec(n_features=10)
    X_test = X_train[100:102]

    for top_k in (3, 5, 15):
        rows = explain(pipe, platt, X_test, spec, top_k=top_k)
        expected = min(top_k, len(spec))
        for r in rows:
            assert len(r.top_k_features) == expected, (
                f"top_k={top_k}: expected {expected} features, got {len(r.top_k_features)}"
            )


def test_spec_v1_labels_coverage():
    """All SPEC_V1 features have a non-empty label and instrument_id."""
    from market_anomaly_detector.features.spec import SPEC_V1

    for s in SPEC_V1:
        assert s.label, f"Empty label for col={s.col}"
        assert s.instrument_id, f"Empty instrument_id for col={s.col}"


def test_permutation_fallback_runs(monkeypatch):
    """When TreeExplainer raises, PermutationExplainer is used and explainer_used=='permutation'."""
    import shap

    import market_anomaly_detector.explain_v2 as ev2

    def _failing_tree(*args, **kwargs):
        raise RuntimeError("forced failure for test")

    monkeypatch.setattr(shap, "TreeExplainer", _failing_tree)

    pipe, platt, X_train = _make_hgb_pipeline_and_platt(n=200, n_features=5)
    spec = _make_feature_spec(n_features=5)
    X_test = X_train[100:103]
    X_background = X_train[:50]

    rows = ev2.explain(pipe, platt, X_test, spec, X_background=X_background)

    assert rows, "explain() returned an empty list"
    for r in rows:
        assert r.explainer_used == "permutation", (
            f"Expected explainer_used='permutation', got {r.explainer_used!r}"
        )


def test_rebuild_fold_deterministic():
    """_rebuild_fold_detect_head called twice on same fold gives bitwise-identical probs."""
    from market_anomaly_detector.eval_harness import make_folds_v2
    from market_anomaly_detector.explain_v2 import _rebuild_fold_detect_head

    rng = np.random.default_rng(0)
    n = 600
    n_features = 10
    weeks = [date(2000, 1, 4) + timedelta(weeks=i) for i in range(n)]
    detect_arr = (rng.random(n) < 0.2).astype(int)
    detect_arr[:30] = 1  # guarantee enough positives

    X_all = rng.standard_normal((n, n_features))
    core_ok = np.ones(n, dtype=bool)
    ku_mask = np.zeros(n, dtype=bool)

    folds = make_folds_v2(weeks, detect_arr, first_test_year=2008, last_test_year=2008)
    if not folds:
        pytest.skip("No folds generated for synthetic data")

    fold = folds[0]

    pipe1, platt1, thr1 = _rebuild_fold_detect_head(X_all, detect_arr, core_ok, ku_mask, fold)
    pipe2, platt2, thr2 = _rebuild_fold_detect_head(X_all, detect_arr, core_ok, ku_mask, fold)

    X_test = X_all[fold.test_lo : fold.test_hi]
    probs1 = platt1.transform(pipe1.decision_function(X_test))
    probs2 = platt2.transform(pipe2.decision_function(X_test))

    assert np.array_equal(probs1, probs2), (
        "Probabilities differ between two identical fold rebuilds"
    )
    assert thr1 == thr2, f"Thresholds differ: {thr1} != {thr2}"
