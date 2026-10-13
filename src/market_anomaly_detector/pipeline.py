from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.ensemble import IsolationForest, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .data import engineer_features, FEATURE_COLS
from .explain import SHAPExplainer

# Augmented feature set: original engineered features + Isolation Forest score
AUG_FEATURE_COLS = FEATURE_COLS + ['iso_score']


class AnomalyPipeline:
    """
    Two-stage anomaly detection pipeline:
      Stage 1 — IsolationForest produces an anomaly score fed forward as a feature.
      Stage 2 — StackingClassifier (DT + RF + LR base, LR meta) trained on the
                 augmented feature set, then calibrated on a held-out val split.
    """

    def __init__(self):
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

        # Calibrate on validation set so predict_proba is meaningful
        self.calibrated = CalibratedClassifierCV(FrozenEstimator(self._stacking), method='sigmoid')
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
        pipeline = joblib.load(path)
        if not isinstance(pipeline, cls):
            raise TypeError(f"Loaded object is {type(pipeline)}, expected AnomalyPipeline.")
        return pipeline
