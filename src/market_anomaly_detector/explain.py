from __future__ import annotations

import numpy as np
import pandas as pd
import shap


class SHAPExplainer:
    """
    Wraps a SHAP TreeExplainer over the Random Forest component of the stacking
    classifier. Explainer is built once at train time; per-prediction SHAP values
    are computed at inference time.
    """

    def __init__(self, stacking_clf, X_background: pd.DataFrame, feature_names: list[str]):
        rf = stacking_clf.named_estimators_['rf']
        self.explainer = shap.TreeExplainer(rf, data=shap.sample(X_background, 100))
        self.feature_names = list(feature_names)

    def top_features(self, X_row: pd.DataFrame, n: int = 5) -> list[dict]:
        """Return the top-n features by absolute SHAP impact for a single row."""
        shap_vals = self.explainer.shap_values(X_row, check_additivity=False)

        # TreeExplainer returns a list [class_0_vals, class_1_vals] for classifiers
        if isinstance(shap_vals, list):
            vals = np.array(shap_vals[1]).flatten()
        else:
            vals = np.array(shap_vals).flatten()

        ranked = sorted(
            zip(self.feature_names, vals.tolist()),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        return [{"feature": f, "impact": round(v, 6)} for f, v in ranked[:n]]
