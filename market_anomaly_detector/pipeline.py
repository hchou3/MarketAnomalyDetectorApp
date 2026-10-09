from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.ensemble import IsolationForest, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, classification_report, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .data import engineer_features, FEATURE_COLS
from .explain import SHAPExplainer

# Augmented feature set: original engineered features + Isolation Forest score
AUG_FEATURE_COLS = FEATURE_COLS + ['iso_score']

_VALID_CALIBRATION = {"auto", "sigmoid", "isotonic"}


class AnomalyPipeline:
    """
    Two-stage anomaly detection pipeline:
      Stage 1 — IsolationForest produces an anomaly score fed forward as a feature.
      Stage 2 — StackingClassifier (DT + RF + LR base, LR meta) trained on the
                 augmented feature set, then calibrated on a held-out val split.
    """

    def __init__(self, calibration: str = "auto"):
        if calibration not in _VALID_CALIBRATION:
            raise ValueError(
                f"calibration must be one of {sorted(_VALID_CALIBRATION)!r}, got {calibration!r}"
            )
        self.calibration = calibration
        self.scaler = StandardScaler()
        self.iso_forest = IsolationForest(
            n_estimators=500,
            contamination='auto',
            max_features=1.0,
            random_state=42,
        )
        self._stacking = StackingClassifier(
            estimators=[
                ('dt', DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42)),
                ('rf', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)),
                ('lr', LogisticRegression(C=0.1, class_weight='balanced', solver='liblinear', max_iter=1000)),
            ],
            final_estimator=LogisticRegression(C=1.0, solver='liblinear', max_iter=1000),
            cv=5,
            passthrough=False,
            n_jobs=-1,
        )
        self.calibrated: CalibratedClassifierCV | None = None
        self.explainer: SHAPExplainer | None = None
        self.calibration_method_: str | None = None
        self.calibration_report_: dict | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scale(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        raw = X[FEATURE_COLS].values
        scaled = self.scaler.fit_transform(raw) if fit else self.scaler.transform(raw)
        return pd.DataFrame(scaled, columns=FEATURE_COLS, index=X.index)

    def _augment(self, X_scaled: pd.DataFrame) -> pd.DataFrame:
        X_aug = X_scaled.copy()
        X_aug['iso_score'] = self.iso_forest.decision_function(X_scaled[FEATURE_COLS])
        return X_aug

    def _oof_calibration_scores(
        self,
        X_v_aug: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict:
        """Compute OOF Brier score and log loss for sigmoid and isotonic on the val split."""
        skf = StratifiedKFold(n_splits=5, shuffle=False)
        frozen = FrozenEstimator(self._stacking)
        report = {}

        for method in ("sigmoid", "isotonic"):
            oof_proba = np.zeros(len(y_val))
            y_arr = np.array(y_val)

            for train_idx, val_idx in skf.split(X_v_aug, y_arr):
                X_fold_tr = X_v_aug.iloc[train_idx]
                y_fold_tr = y_arr[train_idx]
                X_fold_val = X_v_aug.iloc[val_idx]

                cal = CalibratedClassifierCV(frozen, method=method)
                cal.fit(X_fold_tr, y_fold_tr)
                oof_proba[val_idx] = cal.predict_proba(X_fold_val)[:, 1]

            report[method] = {
                "oof_brier": float(brier_score_loss(y_arr, oof_proba)),
                "oof_log_loss": float(log_loss(y_arr, oof_proba)),
            }

        return report

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> None:
        X_tr = self._scale(X_train, fit=True)
        X_v = self._scale(X_val)

        # Stage 1: fit Isolation Forest, append score as feature
        self.iso_forest.fit(X_tr[FEATURE_COLS])
        X_tr_aug = self._augment(X_tr)
        X_v_aug = self._augment(X_v)

        # Stage 2: fit stacking classifier
        self._stacking.fit(X_tr_aug, y_train)

        # Calibration method selection
        self.calibration_report_ = self._oof_calibration_scores(X_v_aug, y_val)

        if self.calibration == "auto":
            sig_brier = self.calibration_report_["sigmoid"]["oof_brier"]
            iso_brier = self.calibration_report_["isotonic"]["oof_brier"]
            # Tie-break in favour of sigmoid (fewer parameters)
            chosen = "sigmoid" if sig_brier <= iso_brier else "isotonic"
        else:
            chosen = self.calibration

        self.calibration_method_ = chosen
        self.calibrated = CalibratedClassifierCV(FrozenEstimator(self._stacking), method=chosen)
        self.calibrated.fit(X_v_aug, y_val)

        # Build SHAP explainer once (expensive) — reused at inference time
        self.explainer = SHAPExplainer(self._stacking, X_tr_aug, AUG_FEATURE_COLS)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _prepare(self, df_features: pd.DataFrame) -> pd.DataFrame:
        X_scaled = self._scale(df_features)
        return self._augment(X_scaled)

    def predict_proba(self, df_features: pd.DataFrame) -> np.ndarray:
        """Accept an already-engineered feature DataFrame."""
        return self.calibrated.predict_proba(self._prepare(df_features))

    def predict(self, df_features: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(df_features)[:, 1] >= 0.5).astype(int)

    def score(self, df: pd.DataFrame) -> dict:
        """
        Public interface. Accepts a raw DataFrame in the same shape as the source
        CSV (needs >=20 rows for rolling features). Returns prediction for the
        last row.

        Returns:
            {
                "anomaly": bool,
                "confidence": float,        # calibrated probability of anomaly
                "top_features": [{"feature": str, "impact": float}, ...]
            }
        """
        if self.calibrated is None:
            raise RuntimeError("Pipeline has not been fitted. Call fit() or load() first.")

        featured = engineer_features(df)
        if featured.empty:
            raise ValueError("Need ≥20 rows to compute rolling features.")

        X_row = featured.iloc[[-1]][FEATURE_COLS]
        X_aug = self._prepare(X_row)

        proba = float(self.calibrated.predict_proba(X_aug)[0, 1])
        anomaly = proba >= 0.5
        top = self.explainer.top_features(X_aug) if self.explainer else []

        return {
            "anomaly": bool(anomaly),
            "confidence": round(proba, 6),
            "top_features": top,
        }

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def evaluate(self, X: pd.DataFrame, y: pd.Series, label: str = "set") -> dict:
        preds = self.predict(X)
        probas = self.predict_proba(X)[:, 1]
        print(f"\n--- {label} ---")
        print(classification_report(y, preds, digits=4))
        return {
            "f1": f1_score(y, preds),
            "predictions": preds,
            "probabilities": probas,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        joblib.dump(self, path)
        print(f"Pipeline saved to {path}")

    @classmethod
    def load(cls, path: str) -> AnomalyPipeline:
        import json
        import warnings
        from pathlib import Path

        pipeline = joblib.load(path)
        if not isinstance(pipeline, cls):
            raise TypeError(f"Loaded object is {type(pipeline)}, expected AnomalyPipeline.")

        sidecar = Path(path).with_suffix(".meta.json")
        if sidecar.exists():
            import sklearn
            import shap as shap_mod
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            trained_sk = meta.get("versions", {}).get("scikit-learn")
            trained_shap = meta.get("versions", {}).get("shap")
            running_sk = sklearn.__version__
            running_shap = shap_mod.__version__
            if trained_sk and trained_sk != running_sk:
                warnings.warn(
                    f"Pipeline was trained with scikit-learn {trained_sk}, "
                    f"running {running_sk}.",
                    stacklevel=2,
                )
            if trained_shap and trained_shap != running_shap:
                warnings.warn(
                    f"Pipeline was trained with shap {trained_shap}, "
                    f"running {running_shap}.",
                    stacklevel=2,
                )

        return pipeline
