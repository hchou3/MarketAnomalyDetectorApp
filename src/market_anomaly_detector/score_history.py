"""Phase 2 Step 2.8 — score history: write weekly regime_scores from 2000 to present.

Usage
-----
    mad-model score-history [--version <v>] [--from <date>]

Scores every W-TUE week that has sufficient feature data, writes two rows per
week (detect head and predict head), and verifies ONNX parity within 1e-5 on
all core-ok rows before writing anything.

Row statuses
------------
detect head : "ok" (scored) | "insufficient_data" (core features missing)
predict head: "experimental" (always — this head has 0/6 advance warnings in
              confirmation and must never be used for alerts)

Idempotent: re-running does not change existing rows.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

from .calibration import PlattCalibrator
from .selection import _core_feature_mask


def _load_artifact(artifact_dir: Path):
    """Load model.joblib, metadata.json, and model.onnx bytes."""
    import joblib

    meta = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    pipe = joblib.load(artifact_dir / "model.joblib")
    onnx_bytes = (artifact_dir / "model.onnx").read_bytes()
    return pipe, meta, onnx_bytes


def _get_active_version(session: "Session") -> str | None:
    from sqlalchemy import select

    from .db.models import ModelVersionRow

    row = session.execute(
        select(ModelVersionRow).where(ModelVersionRow.active.is_(True))
    ).scalar_one_or_none()
    return row.version if row is not None else None


def _existing_score_weeks(session: "Session", version: str) -> set[tuple[date, str]]:
    from sqlalchemy import select

    from .db.models import RegimeScoreRow

    rows = session.execute(
        select(RegimeScoreRow.week, RegimeScoreRow.head).where(
            RegimeScoreRow.model_version == version
        )
    ).all()
    return {(r.week, r.head) for r in rows}


def _weekly_tuesdays(from_date: date, to_date: date) -> list[date]:
    import pandas as pd

    dates = pd.date_range(start=str(from_date), end=str(to_date), freq="W-TUE")
    return [d.date() for d in dates]


def run_score_history(
    session: "Session",
    version: str | None = None,
    from_date: date | None = None,
    artifacts_root: Path | None = None,
    _feature_df=None,
) -> int:
    """Write weekly regime_scores for the active (or specified) model version.

    Parameters
    ----------
    session:
        Active SQLAlchemy session.
    version:
        Model version string. None → uses the active detect version.
    from_date:
        First W-TUE to score. None → starts from 2000-01-04.
    artifacts_root:
        Root directory for artifacts. None → ARTIFACTS_ROOT from final_train.
    _feature_df:
        Pre-built feature DataFrame for testing (bypasses build_from_db).
        Index must be date objects or Timestamps corresponding to W-TUE weeks.
    """
    from sqlalchemy import func, select

    from .db.models import ModelVersionRow, PriceRow, RegimeScoreRow
    from .explain_v2 import explain
    from .features.build import build_from_db
    from .features.spec import SPEC_V1
    from .final_train import ARTIFACTS_ROOT, VAL_END, VAL_START

    # ── 1. Resolve version ────────────────────────────────────────────────────
    if version is None:
        version = _get_active_version(session)
        if version is None:
            print(
                "ERROR: no active model version — run: mad-model activate <version> --head detect",
                file=sys.stderr,
            )
            return 1

    mv_row = session.get(ModelVersionRow, version)
    if mv_row is None:
        print(f"ERROR: version {version!r} not found in model_versions", file=sys.stderr)
        return 1

    # ── 2. Load artifact ──────────────────────────────────────────────────────
    root = artifacts_root if artifacts_root is not None else ARTIFACTS_ROOT
    artifact_dir = root / version

    for path, name in [
        (artifact_dir / "model.joblib", "model.joblib"),
        (artifact_dir / "metadata.json", "metadata.json"),
        (artifact_dir / "model.onnx", "model.onnx"),
    ]:
        if not path.exists():
            print(f"ERROR: artifact {name} not found at {path}", file=sys.stderr)
            return 1

    pipe, meta, onnx_bytes = _load_artifact(artifact_dir)

    platt_detect = PlattCalibrator(use_logit=False)
    platt_detect.a = float(meta["platt_a"])
    platt_detect.b = float(meta["platt_b"])
    thr_detect = float(meta["threshold"])

    predict_meta = meta.get("heads", {}).get("predict", {})
    platt_predict = PlattCalibrator(use_logit=False)
    platt_predict.a = float(predict_meta.get("platt_a", platt_detect.a))
    platt_predict.b = float(predict_meta.get("platt_b", platt_detect.b))
    thr_predict = float(predict_meta.get("threshold", thr_detect))

    feature_order = meta["feature_order"]
    n_features = len(feature_order)

    # ── 3. Build feature weeks ────────────────────────────────────────────────
    if _feature_df is not None:
        feature_df = _feature_df
        tuesdays = [d.date() if hasattr(d, "date") else d for d in feature_df.index]
    else:
        latest_obs = session.execute(select(func.max(PriceRow.obs_date))).scalar_one_or_none()
        if latest_obs is None:
            print("ERROR: no price data in DB", file=sys.stderr)
            return 1

        start = from_date if from_date is not None else date(2000, 1, 4)
        tuesdays = _weekly_tuesdays(start, latest_obs)
        if not tuesdays:
            print("No W-TUE weeks to score.", file=sys.stderr)
            return 0

        print(f"score-history: building features for {len(tuesdays)} weeks …", flush=True)
        feature_df = build_from_db(session, SPEC_V1, tuesdays, as_of=None)

    if feature_df.shape[1] != n_features:
        print(
            f"ERROR: feature_df has {feature_df.shape[1]} columns but model expects {n_features}",
            file=sys.stderr,
        )
        return 1

    X_all = feature_df.values.astype(float)
    core_ok = _core_feature_mask(feature_df, X_all)
    X_core = X_all[core_ok]

    # ── 4. ONNX parity check (must pass before any rows are written) ──────────
    if len(X_core) == 0:
        print(
            "WARNING: no rows with complete core features; writing insufficient_data only.",
            flush=True,
        )
    else:
        try:
            import onnxruntime as ort
        except ImportError:
            print(
                "ERROR: onnxruntime not installed — cannot verify ONNX parity.",
                file=sys.stderr,
            )
            return 1

        try:
            sess_ort = ort.InferenceSession(onnx_bytes)
            input_name = sess_ort.get_inputs()[0].name
            ort_probs = (
                sess_ort.run(["calibrated_probability"], {input_name: X_core})[0]
                .flatten()
                .astype(np.float64)
            )
        except Exception as exc:
            print(f"ERROR: ONNX inference failed: {exc}", file=sys.stderr)
            return 1

        raw_core = pipe.decision_function(X_core)
        sklearn_probs = platt_detect.transform(raw_core).astype(np.float64)
        max_diff = float(np.max(np.abs(ort_probs - sklearn_probs)))
        if max_diff > 1e-5:
            print(
                f"ERROR: ONNX parity check failed (max_diff={max_diff:.2e} > 1e-5). "
                "Not writing any scores.",
                file=sys.stderr,
            )
            return 1
        print(f"score-history: ONNX parity OK (max_diff={max_diff:.2e})", flush=True)

    # ── 5. Score all weeks ────────────────────────────────────────────────────
    raw_all = np.full(len(tuesdays), np.nan)
    if len(X_core) > 0:
        raw_all[core_ok] = pipe.decision_function(X_core)

    cal_detect = np.full(len(tuesdays), np.nan)
    if len(X_core) > 0:
        cal_detect[core_ok] = platt_detect.transform(raw_all[core_ok])

    cal_predict = np.full(len(tuesdays), np.nan)
    if len(X_core) > 0:
        cal_predict[core_ok] = platt_predict.transform(raw_all[core_ok])

    # ── 6. SHAP top-5 (detect head, core_ok rows only) ───────────────────────
    spec_by_col = {s.col: s for s in SPEC_V1}
    df_cols = list(feature_df.columns)
    feature_spec_for_explain = [spec_by_col[c] for c in df_cols if c in spec_by_col]

    top_features_all: list[str | None] = [None] * len(tuesdays)
    if (
        len(X_core) > 0
        and feature_spec_for_explain
        and len(feature_spec_for_explain) == X_core.shape[1]
    ):
        try:
            core_weeks = [tuesdays[i] for i in range(len(tuesdays)) if core_ok[i]]
            explain_rows = explain(
                pipe,
                platt_detect,
                X_core,
                feature_spec_for_explain,
                weeks=core_weeks,
                top_k=5,
            )
            core_indices = [i for i in range(len(tuesdays)) if core_ok[i]]
            for idx, er in zip(core_indices, explain_rows):
                top_features_all[idx] = json.dumps(
                    [
                        {
                            "feature": tf.col,
                            "label": tf.label,
                            "instrument": spec_by_col[tf.col].instrument_id
                            if tf.col in spec_by_col
                            else "",
                            "contribution": round(tf.contribution, 6),
                            "value": round(tf.feature_value, 6),
                        }
                        for tf in er.top_k_features
                    ]
                )
        except Exception as exc:
            print(
                f"WARNING: SHAP computation failed ({exc}); top_features will be NULL.",
                flush=True,
            )

    # ── 7. Idempotent upsert ─────────────────────────────────────────────────
    existing = _existing_score_weeks(session, version)
    now = datetime.now(timezone.utc)
    n_new_detect = n_new_predict = 0
    n_insuff = 0

    for i, week in enumerate(tuesdays):
        is_core = bool(core_ok[i])
        if not is_core:
            n_insuff += 1

        if (week, "detect") not in existing:
            status_detect = "ok" if is_core else "insufficient_data"
            prob_detect = float(cal_detect[i]) if is_core else None
            alert_detect = bool(cal_detect[i] >= thr_detect) if is_core else None
            session.add(
                RegimeScoreRow(
                    week=week,
                    model_version=version,
                    head="detect",
                    probability=prob_detect,
                    alert=alert_detect,
                    threshold=thr_detect,
                    status=status_detect,
                    top_features=top_features_all[i] if is_core else None,
                    created_at=now,
                )
            )
            n_new_detect += 1

        if (week, "predict") not in existing:
            session.add(
                RegimeScoreRow(
                    week=week,
                    model_version=version,
                    head="predict",
                    probability=float(cal_predict[i]) if is_core else None,
                    alert=None,  # predict head never fires alerts
                    threshold=thr_predict,
                    status="experimental",
                    top_features=None,
                    created_at=now,
                )
            )
            n_new_predict += 1

    session.commit()

    # ── 8. Summary ────────────────────────────────────────────────────────────
    total = len(tuesdays)
    n_scored = int(core_ok.sum())
    alert_weeks = sum(
        1
        for i in range(len(tuesdays))
        if core_ok[i] and not np.isnan(cal_detect[i]) and cal_detect[i] >= thr_detect
    )

    in_sample_wks = [w for w in tuesdays if w < VAL_START]
    cal_window_wks = [w for w in tuesdays if VAL_START <= w <= VAL_END]
    unvalidated_wks = [w for w in tuesdays if w > VAL_END]

    sep = "─" * 60
    print(
        f"\n{sep}\n"
        f"score-history: {version}\n"
        f"{sep}\n"
        f"  Weeks total:          {total}\n"
        f"  Weeks scored (ok):    {n_scored}\n"
        f"  insufficient_data:    {n_insuff}\n"
        f"  Alert weeks (detect): {alert_weeks}\n"
        f"  New rows (detect):    {n_new_detect}\n"
        f"  New rows (predict):   {n_new_predict}\n"
        f"\n"
        f"  Periods:\n"
        f"    In-sample (before {VAL_START}): {len(in_sample_wks)} weeks\n"
        f"    *** In-sample scores are NOT evidence of performance.  ***\n"
        f"    *** The out-of-sample record is the walk-forward and   ***\n"
        f"    *** confirmation reports.                              ***\n"
        f"\n"
        f"    Calibration window ({VAL_START}–{VAL_END}): {len(cal_window_wks)} weeks\n"
        f"    Unvalidated (after {VAL_END}): {len(unvalidated_wks)} weeks\n"
        f"    (Unvalidated scores have no labels; use for context only.\n"
        f"     Decision L1 from architecture plan applies.)\n"
        f"{sep}"
    )
    return 0
