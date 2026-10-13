from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def _class1_values(shap_vals) -> np.ndarray:
    """Normalise SHAP output to a 1-D array of class-1 contributions (one per feature).

    Handles three shapes produced by different SHAP versions:
      - list [class_0_array, class_1_array]   → take element [1], then row 0
      - 3-D ndarray (n_rows, n_features, n_classes) → take [..., 1], then row 0
      - 2-D ndarray (n_rows, n_features)       → use as-is, then row 0
    """
    if isinstance(shap_vals, list):
        arr = np.array(shap_vals[1])
    else:
        arr = np.array(shap_vals)
        if arr.ndim == 3:
            arr = arr[..., 1]

    # arr is now (n_rows, n_features) or (n_features,)
    if arr.ndim == 2:
        arr = arr[0]
    return arr


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
        vals = _class1_values(shap_vals)

        if len(vals) != len(self.feature_names):
            raise ValueError(
                f"SHAP values length ({len(vals)}) does not match "
                f"feature_names length ({len(self.feature_names)})"
            )

        ranked = sorted(
            zip(self.feature_names, vals.tolist(), strict=True),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        return [{"feature": f, "impact": round(v, 6)} for f, v in ranked[:n]]
