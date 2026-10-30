"""Phase 2 Step 2.7 — final production fit, ONNX export, and model registry.

`mad-model train --final` fits the frozen v1 HGB on all labelled history up to
the validation window, exports to ONNX, writes the artifact set, and inserts an
inactive row into model_versions.

`mad-model export <version>` re-exports ONNX from an existing joblib artifact.

`mad-model activate <version>` marks a version active in the model_versions
table (detect head only for v1; predict head remains experimental).

Artifact layout
---------------
artifacts/<model_version>/
  model.joblib   sklearn HGB pipeline (detect head)
  model.onnx     HGB + Platt graph; input [N,51] float32, output calibrated_probability
  metadata.json  all fields required by the spec
  train_X.npy    first 500 training rows kept for offline export/parity re-check

model_version format: v1-<YYYY-MM-DD>-<spec_hash[:8]>
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

from .calibration import PlattCalibrator, onnx_export_with_platt
from .eval_harness import KNOWN_UNLABELED, choose_threshold
from .features.spec import SPEC_V1
from .features.spec import spec_hash as _spec_hash
from .selection import (
    FAR_BUDGET,
    HORIZON,
    _core_feature_mask,
    check_onnx_gate,
    make_hgb_pipeline,
)

# ── Frozen constants ──────────────────────────────────────────────────────────

# SHA-256 of SPEC_V1; change here if SPEC_V1 is updated intentionally.
PINNED_SPEC_HASH: str = "b8c2d65c449392a806bf30c8fc596785cab943aa08268fae4fff593b14ab8a3f"

# SHA-256 of FROZEN_CONFIG in confirm.py (331acff8 prefix matches the report).
PINNED_CONFIG_HASH: str = "331acff81da50cf5180a53b490b10764daafde0522aaa6a5bda02d0e45ad2c01"

# Final production validation window: last two labelled years.
# VAL_START is the first W-TUE on or after (last_label - 2 years).
VAL_START: date = date(2019, 4, 23)
VAL_END: date = date(2021, 4, 20)  # last labelled week in ews_v1

# Default artifact root (git-ignored).
_REPO_ROOT = Path(__file__).parent.parent.parent
ARTIFACTS_ROOT: Path = _REPO_ROOT / "artifacts"

# ── Metrics from committed reports ────────────────────────────────────────────
# Copied verbatim from docs/reports/ — not recomputed here.

_FINAL_DEV_METRICS: dict = {
    "event_f1": 0.527,
    "row_f1": 0.429,
    "pr_auc": 0.571,
    "brier": 0.148,
    "source": "2026-10-29-model-selection.md",
}

_FINAL_CONFIRM_METRICS: dict = {
    "event_f1": 0.645,
    "episodes_caught": "4/6",
    "row_f1": 0.604,
    "pr_auc": 0.612,
    "brier": 0.101,
    "log_loss": 0.408,
    "fa_total": 3,
    "fa_per_year": 0.75,
    "predict_advance_warnings": "0/6",
    "source": "2026-10-21-final-confirmation-331acff8.md",
}


# ── Core fit logic ────────────────────────────────────────────────────────────


def _fit_final_model(
    X_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    out_dir: Path,
    label_set: str = "ews_v1",
    *,
    far_budget: float = FAR_BUDGET,
    val_start: date = VAL_START,
    val_end: date = VAL_END,
) -> tuple[str, Path]:
    """Fit HGB detect + predict heads on historical data and export artifacts.

    Parameters
    ----------
    X_df:
        Feature DataFrame; columns must match SPEC_V1 col names in order.
        DatetimeIndex or positional index; rows aligned with labels_df.
    labels_df:
        DataFrame with 'week' (date) and 'value' (0/1) columns.
    out_dir:
        Root directory for artifacts (default: ARTIFACTS_ROOT).
    label_set:
        Label set identifier for metadata.
    far_budget:
        False-alarm rate budget for threshold selection (default: 5%).
    val_start, val_end:
        Validation window boundaries.  Defaults to the production window;
        override for testing with synthetic data.

    Returns
    -------
    (model_version, artifact_dir)
    """
    import importlib.metadata
    import io

    import joblib

    # ── Spec hash guard ───────────────────────────────────────────────────────
    current_hash = _spec_hash(SPEC_V1)
    if current_hash != PINNED_SPEC_HASH:
        raise RuntimeError(
            f"SPEC_V1 hash mismatch: current={current_hash[:8]}, "
            f"pinned={PINNED_SPEC_HASH[:8]}. "
            "Update PINNED_SPEC_HASH in final_train.py if this is intentional."
        )

    # ── Setup ─────────────────────────────────────────────────────────────────
    all_weeks: list[date] = list(labels_df["week"])
    detect_arr = labels_df["value"].values.astype(int)
    n_total = len(all_weeks)

    # Find train/val boundary (first row on or after val_start)
    n_train = next(
        (i for i, w in enumerate(all_weeks) if w >= val_start),
        n_total,
    )
    n_val = n_total - n_train
    if n_val < 4:
        raise ValueError(
            f"Validation window has only {n_val} rows (need >= 4). "
            f"Check that data covers {val_start} to {val_end}."
        )

    val_positives = int((detect_arr[n_train:] == 1).sum())
    if val_positives < 8:
        raise ValueError(f"Validation window has only {val_positives} positive labels (need >= 8).")

    X_all = X_df.values.astype(float)
    n_features = X_all.shape[1]
    core_ok = _core_feature_mask(X_df, X_all)

    ku_mask = np.array(
        [any(ks <= w <= ke for ks, ke, _ in KNOWN_UNLABELED) for w in all_weeks],
        dtype=bool,
    )

    # ── Split ─────────────────────────────────────────────────────────────────
    X_train_raw = X_all[:n_train]
    X_val_raw = X_all[n_train:]
    y_train_det_raw = detect_arr[:n_train]
    y_val_det_raw = detect_arr[n_train:]
    core_train = core_ok[:n_train]
    core_val = core_ok[n_train:]
    val_ku = ku_mask[n_train:]

    X_train = X_train_raw[core_train]
    y_train_det = y_train_det_raw[core_train]
    X_val = X_val_raw[core_val]
    y_val_det = y_val_det_raw[core_val]
    val_ku_filtered = val_ku[core_val]

    # ── Detect head ───────────────────────────────────────────────────────────
    pipe_detect = make_hgb_pipeline()
    pipe_detect.fit(X_train, y_train_det)

    val_raw_detect = pipe_detect.decision_function(X_val)
    platt_detect = PlattCalibrator(use_logit=False).fit(val_raw_detect, y_val_det)
    val_cal_detect = platt_detect.transform(val_raw_detect)
    thr_detect = choose_threshold(
        y_val_det, val_cal_detect, far_budget, excluded_mask=val_ku_filtered
    )

    # ── Predict head (experimental) ───────────────────────────────────────────
    from .labels import build_targets

    targets = build_targets(labels_df, horizon=HORIZON)
    predict_arr = targets["predict"].values

    # Boundary purge: last HORIZON train rows have predict[t] reading val labels.
    train_pred_mask = ~np.isnan(predict_arr[:n_train])
    train_purge_from = max(0, n_train - HORIZON)
    n_train_purged = int(train_pred_mask[train_purge_from:].sum())
    train_pred_mask[train_purge_from:] = False

    # build_targets already sets NaN for the last HORIZON rows of the full set,
    # so no additional purge is needed at the val/end boundary.
    val_pred_mask = ~np.isnan(predict_arr[n_train:])

    y_train_pred = predict_arr[:n_train][core_train & train_pred_mask].astype(int)
    X_train_pred = X_train_raw[core_train & train_pred_mask]
    y_val_pred = predict_arr[n_train:][core_val & val_pred_mask].astype(int)
    X_val_pred = X_val_raw[core_val & val_pred_mask]
    val_ku_pred = val_ku[core_val & val_pred_mask]

    print(
        f"  [predict] boundary purge: {n_train_purged} train rows excluded "
        f"(last {HORIZON} rows before val_start)"
    )

    platt_predict: PlattCalibrator | None = None
    thr_predict = 0.5
    if len(y_train_pred) >= 20 and len(np.unique(y_train_pred)) >= 2:
        pipe_predict = make_hgb_pipeline()
        pipe_predict.fit(X_train_pred, y_train_pred)

        if len(y_val_pred) >= 4 and len(np.unique(y_val_pred)) >= 2:
            val_raw_pred = pipe_predict.decision_function(X_val_pred)
            platt_predict = PlattCalibrator(use_logit=False).fit(val_raw_pred, y_val_pred)
            val_cal_pred = platt_predict.transform(val_raw_pred)
            thr_predict = choose_threshold(
                y_val_pred, val_cal_pred, far_budget, excluded_mask=val_ku_pred
            )
        else:
            platt_predict = PlattCalibrator(use_logit=False)
            platt_predict.a, platt_predict.b = 1.0, 0.0
    else:
        platt_predict = PlattCalibrator(use_logit=False)
        platt_predict.a, platt_predict.b = 1.0, 0.0

    # ── ONNX parity on training rows ─────────────────────────────────────────
    gate_X = X_train[: min(500, len(X_train))].astype(np.float32)
    gate = check_onnx_gate("hgb", pipe_detect, platt_detect, gate_X, n_features)
    if not gate.passed:
        raise RuntimeError(
            f"ONNX parity check failed (max_diff={gate.max_diff}, "
            f"error={gate.error}). Refusing to write artifacts."
        )

    onnx_bytes = onnx_export_with_platt(pipe_detect, platt_detect, n_features, X_sample=gate_X[:5])

    # ── Model version and output directory ────────────────────────────────────
    model_version = f"v1-{date.today().isoformat()}-{PINNED_SPEC_HASH[:8]}"
    artifact_dir = out_dir / model_version
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # ── Write artifacts ───────────────────────────────────────────────────────
    joblib.dump(pipe_detect, artifact_dir / "model.joblib")
    (artifact_dir / "model.onnx").write_bytes(onnx_bytes)
    # Keep first 500 training rows for offline parity re-checks.
    np.save(artifact_dir / "train_X.npy", gate_X)

    # ── Feature frame hash ────────────────────────────────────────────────────
    buf = io.BytesIO()
    np.save(buf, X_all)
    frame_hash = hashlib.sha256(buf.getvalue()).hexdigest()

    # ── Library versions ──────────────────────────────────────────────────────
    lib_versions: dict[str, str] = {}
    for pkg in ("scikit-learn", "skl2onnx", "onnxruntime", "numpy", "scipy", "pandas"):
        try:
            lib_versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            lib_versions[pkg] = "unknown"

    # ── Metadata ──────────────────────────────────────────────────────────────
    feature_order = [f.col for f in SPEC_V1]
    feature_labels = {f.col: f.label for f in SPEC_V1}

    metadata: dict = {
        "version": model_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "hgb",
        "spec_hash": PINNED_SPEC_HASH,
        "config_hash": PINNED_CONFIG_HASH,
        "feature_order": feature_order,
        "feature_labels": feature_labels,
        "label_set": label_set,
        "horizon": HORIZON,
        "training_range": {
            "start": str(all_weeks[0]) if all_weeks else None,
            "end": str(all_weeks[n_train - 1]) if n_train > 0 else None,
            "n_rows_raw": n_train,
            "n_rows_filtered": int(core_train.sum()),
        },
        "val_range": {
            "start": str(val_start),
            "end": str(val_end),
            "n_rows_raw": n_val,
            "n_rows_filtered": int(core_val.sum()),
        },
        "feature_frame_hash": frame_hash,
        "library_versions": lib_versions,
        "platt_a": platt_detect.a,
        "platt_b": platt_detect.b,
        "threshold": thr_detect,
        "onnx_parity_max_diff": gate.max_diff,
        "explainer": "tree, log-odds",
        "dev_metrics": _FINAL_DEV_METRICS,
        "confirm_metrics": _FINAL_CONFIRM_METRICS,
        "heads": {
            "detect": {
                "role": "serving",
                "platt_a": platt_detect.a,
                "platt_b": platt_detect.b,
                "threshold": thr_detect,
            },
            "predict": {
                "role": "experimental",
                "platt_a": platt_predict.a,
                "platt_b": platt_predict.b,
                "threshold": thr_predict,
                "note": "0/6 advance warnings in confirmation — not activated",
            },
        },
    }

    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return model_version, artifact_dir


# ── DB registration ───────────────────────────────────────────────────────────


def _register_version(
    session: "Session",
    model_version: str,
    artifact_dir: Path,
    label_set: str,
    metadata: dict,
) -> None:
    """Insert an inactive row into model_versions."""
    from .db.models import ModelVersionRow

    existing = session.get(ModelVersionRow, model_version)
    if existing is not None:
        print(f"  [registry] {model_version} already exists — skipping insert")
        return

    row = ModelVersionRow(
        version=model_version,
        created_at=datetime.now(timezone.utc),
        label_set=label_set,
        spec_hash=PINNED_SPEC_HASH,
        config_hash=PINNED_CONFIG_HASH,
        heads=json.dumps(metadata["heads"]),
        metrics=json.dumps(
            {"dev": metadata["dev_metrics"], "confirm": metadata["confirm_metrics"]}
        ),
        threshold=float(metadata["threshold"]),
        platt_a=float(metadata["platt_a"]),
        platt_b=float(metadata["platt_b"]),
        artifact_path=str(artifact_dir),
        active=False,
    )
    session.add(row)
    session.commit()


# ── Public entry points ───────────────────────────────────────────────────────


def run_final_train(
    session: "Session | None",
    out_dir: Path | None = None,
    label_set: str = "ews_v1",
) -> int:
    """Load data from DB, fit the final model, export artifacts, register version."""
    from .features.build import build_from_db
    from .labels import load_labels

    if session is None:
        print("ERROR: database session required for train --final", file=sys.stderr)
        return 1

    labels_df = load_labels(session, label_set=label_set)
    if labels_df.empty:
        print(f"ERROR: no labels for {label_set!r}", file=sys.stderr)
        return 1

    weeks: list[date] = list(labels_df["week"])
    feature_df = build_from_db(session, SPEC_V1, weeks, as_of=None)

    artifacts_root = out_dir if out_dir is not None else ARTIFACTS_ROOT

    try:
        model_version, artifact_dir = _fit_final_model(
            feature_df, labels_df, artifacts_root, label_set=label_set
        )
    except (ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    meta = _load_metadata(artifact_dir)
    _register_version(session, model_version, artifact_dir, label_set, meta)

    if artifact_dir.is_relative_to(_REPO_ROOT):
        rel = artifact_dir.relative_to(_REPO_ROOT)
    else:
        rel = artifact_dir
    print(f"train: {model_version} -> {rel}")
    return 0


def run_export(
    version: str,
    artifacts_root: Path | None = None,
    session: "Session | None" = None,
) -> int:
    """Re-export ONNX from an existing version's joblib + update metadata.json."""
    import importlib.metadata

    import joblib

    root = artifacts_root if artifacts_root is not None else ARTIFACTS_ROOT
    artifact_dir = root / version
    joblib_path = artifact_dir / "model.joblib"
    meta_path = artifact_dir / "metadata.json"
    train_X_path = artifact_dir / "train_X.npy"

    if not joblib_path.exists():
        print(f"ERROR: {joblib_path} not found", file=sys.stderr)
        return 1
    if not meta_path.exists():
        print(f"ERROR: {meta_path} not found", file=sys.stderr)
        return 1

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    pipe = joblib.load(joblib_path)

    platt = PlattCalibrator(use_logit=False)
    platt.a = float(meta["platt_a"])
    platt.b = float(meta["platt_b"])

    n_features = len(meta["feature_order"])

    gate_X: np.ndarray
    if train_X_path.exists():
        gate_X = np.load(train_X_path).astype(np.float32)
    else:
        gate_X = np.zeros((5, n_features), dtype=np.float32)

    gate = check_onnx_gate("hgb", pipe, platt, gate_X, n_features)
    if not gate.passed:
        print(
            f"ERROR: ONNX parity check failed (max_diff={gate.max_diff}, "
            f"error={gate.error}). Not overwriting model.onnx.",
            file=sys.stderr,
        )
        return 1

    onnx_bytes = onnx_export_with_platt(pipe, platt, n_features, X_sample=gate_X[:5])
    (artifact_dir / "model.onnx").write_bytes(onnx_bytes)

    # Refresh library versions and parity diff in metadata.
    lib_versions: dict[str, str] = {}
    for pkg in ("scikit-learn", "skl2onnx", "onnxruntime", "numpy", "scipy", "pandas"):
        try:
            lib_versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            lib_versions[pkg] = "unknown"
    meta["library_versions"] = lib_versions
    meta["onnx_parity_max_diff"] = gate.max_diff
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"export: {version} -> max_diff={gate.max_diff:.2e}")
    return 0


def run_activate(
    version: str,
    head: str,
    session: "Session",
) -> int:
    """Activate a model version (detect head only for v1)."""
    from sqlalchemy import update

    from .db.models import ModelVersionRow

    if head != "detect":
        print(
            f"ERROR: only 'detect' head can be activated in v1 "
            f"(got {head!r}). predict head is permanently experimental.",
            file=sys.stderr,
        )
        return 1

    row = session.get(ModelVersionRow, version)
    if row is None:
        print(f"ERROR: version {version!r} not found in model_versions", file=sys.stderr)
        return 1

    # Deactivate all versions, then activate the target.
    session.execute(update(ModelVersionRow).values(active=False))
    session.execute(
        update(ModelVersionRow).where(ModelVersionRow.version == version).values(active=True)
    )
    session.commit()

    print(f"activate: {version} (head={head}) -> active")
    return 0


def _load_metadata(artifact_dir: Path) -> dict:
    meta_path = artifact_dir / "metadata.json"
    return json.loads(meta_path.read_text(encoding="utf-8"))
