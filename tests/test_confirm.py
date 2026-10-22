"""Synthetic mechanics tests for confirm.py (Step 2.5).

Tests cover:
- config_hash changes when any constant changes
- run_once guard blocks re-runs and allows force
- confirmation folds cover only 2018–2021 test weeks
- 2021 predict-head rows after the last label are dropped
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from market_anomaly_detector.confirm import (
    CONFIRM_FIRST_YEAR,
    CONFIRM_LAST_YEAR,
    ConfirmConfig,
    config_hash,
    run_once_check,
)

# ── Config hash ───────────────────────────────────────────────────────────────


def test_config_hash_is_deterministic():
    """Same config always produces the same hash."""
    cfg = ConfirmConfig()
    assert config_hash(cfg) == config_hash(cfg)
    assert config_hash(cfg) == config_hash(ConfirmConfig())


def test_config_hash_changes_when_model_changes():
    cfg1 = ConfirmConfig()
    cfg2 = ConfirmConfig(model="logreg")
    assert config_hash(cfg1) != config_hash(cfg2)


def test_config_hash_changes_when_hgb_depth_changes():
    cfg1 = ConfirmConfig()
    cfg2 = ConfirmConfig(hgb_max_depth=5)
    assert config_hash(cfg1) != config_hash(cfg2)


def test_config_hash_changes_when_far_budget_changes():
    cfg1 = ConfirmConfig()
    cfg2 = ConfirmConfig(far_budget=0.10)
    assert config_hash(cfg1) != config_hash(cfg2)


def test_config_hash_changes_when_horizon_changes():
    cfg1 = ConfirmConfig()
    cfg2 = ConfirmConfig(horizon=8)
    assert config_hash(cfg1) != config_hash(cfg2)


def test_config_hash_changes_when_year_range_changes():
    cfg1 = ConfirmConfig()
    cfg2 = ConfirmConfig(last_test_year=2022)
    assert config_hash(cfg1) != config_hash(cfg2)


def test_config_hash_is_hex_string():
    h = config_hash(ConfirmConfig())
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


# ── Run-once guard ────────────────────────────────────────────────────────────


def test_run_once_guard_allows_first_run(tmp_path):
    """No existing report → guard passes silently."""
    cfg = ConfirmConfig()
    h = config_hash(cfg)
    run_once_check(tmp_path, h, force=False, reason="")


def test_run_once_guard_blocks_rerun(tmp_path):
    """Existing report with same hash → guard raises SystemExit."""
    cfg = ConfirmConfig()
    h = config_hash(cfg)
    (tmp_path / f"2026-01-01-final-confirmation-{h[:8]}.md").write_text("mock report")
    with pytest.raises(SystemExit, match="already exists"):
        run_once_check(tmp_path, h, force=False, reason="")


def test_run_once_guard_allows_force_with_reason(tmp_path):
    """Existing report + --force + --reason → guard passes (prints warning)."""
    cfg = ConfirmConfig()
    h = config_hash(cfg)
    (tmp_path / f"2026-01-01-final-confirmation-{h[:8]}.md").write_text("mock report")
    run_once_check(tmp_path, h, force=True, reason="replicating for audit")


def test_run_once_guard_different_hash_allowed(tmp_path):
    """Report with different hash does not block a new config."""
    cfg1 = ConfirmConfig()
    cfg2 = ConfirmConfig(hgb_max_depth=5)
    h1 = config_hash(cfg1)
    h2 = config_hash(cfg2)
    assert h1 != h2
    (tmp_path / f"2026-01-01-final-confirmation-{h1[:8]}.md").write_text("mock report")
    # Different config hash → no block
    run_once_check(tmp_path, h2, force=False, reason="")


# ── Confirmation folds cover only 2018–2021 ──────────────────────────────────


def _make_weekly_labels(start: date, n: int, anomaly_period_frac: float = 0.15) -> pd.DataFrame:
    """Synthetic weekly labels from *start* for *n* weeks."""
    weeks = [start + timedelta(weeks=i) for i in range(n)]
    rng = np.random.default_rng(0)
    values = (rng.random(n) < anomaly_period_frac).astype(int)
    # Ensure enough positives throughout for folds to work
    for i in range(0, n, 50):
        values[i] = 1
    return pd.DataFrame({"week": weeks, "value": values})


def test_confirm_folds_test_weeks_in_range():
    """make_folds_v2 with first=2018 last=2021 → all test weeks are 2018–2021."""
    from market_anomaly_detector.eval_harness import make_folds_v2

    labels_df = _make_weekly_labels(date(2000, 1, 4), 1150)
    weeks = list(labels_df["week"])
    detect = labels_df["value"].values.astype(int)

    folds = make_folds_v2(weeks, detect, first_test_year=2018, last_test_year=2021)

    assert len(folds) == 4, f"Expected 4 folds (one per year), got {len(folds)}"
    for fold in folds:
        test_weeks = weeks[fold.test_lo : fold.test_hi]
        assert all(CONFIRM_FIRST_YEAR <= w.year <= CONFIRM_LAST_YEAR for w in test_weeks), (
            f"Fold {fold.test_year}: test weeks outside 2018–2021"
        )


def test_confirm_folds_test_year_matches():
    """Each fold's test_year matches the year of its test weeks."""
    from market_anomaly_detector.eval_harness import make_folds_v2

    labels_df = _make_weekly_labels(date(2000, 1, 4), 1150)
    weeks = list(labels_df["week"])
    detect = labels_df["value"].values.astype(int)

    folds = make_folds_v2(weeks, detect, first_test_year=2018, last_test_year=2021)
    for fold in folds:
        test_weeks = weeks[fold.test_lo : fold.test_hi]
        assert all(w.year == fold.test_year for w in test_weeks)


# ── 2021 predict-head rows after last label are dropped ───────────────────────


def test_predict_head_drops_last_h_rows():
    """build_targets sets last h=4 rows to NaN predict; those rows are excluded
    from predict head evaluation (test_pred_mask filters them out)."""
    from market_anomaly_detector.labels import build_targets

    horizon = 4
    n = 20
    weeks = [date(2021, 1, 5) + timedelta(weeks=i) for i in range(n)]
    values = np.zeros(n, dtype=int)
    values[3] = 1
    values[7] = 1

    labels_df = pd.DataFrame({"week": weeks, "value": values})
    targets = build_targets(labels_df, horizon=horizon)
    pred = targets["predict"].values

    assert np.all(np.isnan(pred[-horizon:])), "Last h rows must be NaN"
    assert np.all(~np.isnan(pred[:-horizon])), "Rows before last h must not be NaN"


def test_predict_head_rows_excluded_in_fold_eval():
    """When the fold test window ends at the last label, the predict
    test mask correctly excludes the final h rows."""
    from market_anomaly_detector.labels import build_targets

    horizon = 4
    weeks = [date(2021, 1, 5) + timedelta(weeks=i) for i in range(16)]
    values = np.zeros(len(weeks), dtype=int)
    values[2] = 1
    values[5] = 1

    labels_df = pd.DataFrame({"week": weeks, "value": values})
    targets = build_targets(labels_df, horizon=horizon)
    predict_arr = targets["predict"].values

    # Simulate the fold: test window is the full 16-week range
    test_pred_mask = ~np.isnan(predict_arr)
    # Exactly the last h rows should be excluded
    assert test_pred_mask[:-horizon].all()
    assert not test_pred_mask[-horizon:].any()
    # The number of valid predict rows = n - h
    assert test_pred_mask.sum() == len(weeks) - horizon


# ── Frozen config dataclass ───────────────────────────────────────────────────


def test_frozen_config_is_immutable():
    """ConfirmConfig is frozen — mutation raises FrozenInstanceError."""
    cfg = ConfirmConfig()
    with pytest.raises(Exception):  # FrozenInstanceError
        cfg.model = "other"  # type: ignore[misc]


def test_frozen_config_default_values():
    """FROZEN_CONFIG default values match the spec."""
    from market_anomaly_detector.confirm import FROZEN_CONFIG

    assert FROZEN_CONFIG.model == "hgb"
    assert FROZEN_CONFIG.hgb_max_depth == 3
    assert FROZEN_CONFIG.hgb_learning_rate == 0.05
    assert FROZEN_CONFIG.hgb_max_iter == 200
    assert FROZEN_CONFIG.far_budget == 0.05
    assert FROZEN_CONFIG.horizon == 4
    assert FROZEN_CONFIG.merge_gap == 2
    assert FROZEN_CONFIG.first_test_year == 2018
    assert FROZEN_CONFIG.last_test_year == 2021
