"""Platt (sigmoid) calibration with explicit (a, b) parameters.

PlattCalibrator fits by minimising binary cross-entropy via L-BFGS-B.
onnx_export_with_platt() converts a fitted sklearn pipeline + Platt calibration
to ONNX bytes for inference-time deployment.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator

# Guards for idempotent registration
_CONVERTERS_REGISTERED = False


class PlattCalibrator(BaseEstimator):
    """Sigmoid calibration: P(y=1|score) = sigmoid(a * z + b).

    When use_logit=True (default): z = logit(clip(score, 1e-6, 1-1e-6)) — for
    probability inputs in (0, 1) such as predict_proba outputs.
    When use_logit=False: z = score — for raw margin / decision_function inputs
    (logreg, HGB) where the score is already in log-odds space.

    Fitted by minimising binary cross-entropy (NLL) on (scores, labels).
    """

    def __init__(self, use_logit: bool = True) -> None:
        self.use_logit = use_logit
        self.a: float | None = None
        self.b: float | None = None

    @staticmethod
    def _logodds_of_proba(scores: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def _to_input(self, scores: np.ndarray) -> np.ndarray:
        z = np.asarray(scores, dtype=float)
        return self._logodds_of_proba(z) if self.use_logit else z

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> PlattCalibrator:
        yz = self._to_input(scores)
        yt = np.asarray(labels, dtype=float)

        def nll(params: np.ndarray) -> float:
            a, b = params
            p = np.clip(expit(a * yz + b), 1e-15, 1 - 1e-15)
            return -float(np.mean(yt * np.log(p) + (1 - yt) * np.log(1 - p)))

        res = minimize(nll, [1.0, 0.0], method="L-BFGS-B")
        self.a, self.b = float(res.x[0]), float(res.x[1])
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if self.a is None:
            raise RuntimeError("Call fit() before transform().")
        return expit(self.a * self._to_input(scores) + self.b)

    def fit_transform(self, scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return self.fit(scores, labels).transform(scores)


def _apply_skl2onnx_shims() -> None:
    """Apply version-specific compatibility patches to skl2onnx before conversion."""
    import importlib.metadata

    try:
        ver = importlib.metadata.version("skl2onnx")
    except importlib.metadata.PackageNotFoundError:
        return

    if ver.startswith("1.20."):
        # skl2onnx 1.20 passes np.uint8 for nodes_missing_value_tracks_true; onnx 1.23 rejects
        # non-int. Fixed upstream but not in 1.20.x.
        import skl2onnx.common.tree_ensemble as te

        _orig = te.add_node

        def _add_node_shim(*args, nodes_missing_value_tracks_true=0, **kwargs):
            return _orig(
                *args,
                nodes_missing_value_tracks_true=int(nodes_missing_value_tracks_true),
                **kwargs,
            )

        te.add_node = _add_node_shim


def _register_converters() -> None:
    """Register skl2onnx converters for MedianImputeWithIndicator and IsoForestAppender.

    Uses lazy import (called inside onnx_export_with_platt) to avoid circular imports
    between calibration.py and selection.py.  Idempotent: subsequent calls are no-ops.
    """
    global _CONVERTERS_REGISTERED
    if _CONVERTERS_REGISTERED:
        return

    try:
        import onnx as _onnx
        from skl2onnx import update_registered_converter
        from skl2onnx.algebra.onnx_operator import OnnxSubEstimator
        from skl2onnx.algebra.onnx_ops import OnnxCast, OnnxConcat, OnnxIsNaN, OnnxWhere
        from skl2onnx.common.data_types import FloatTensorType

        # Lazy to avoid circular: calibration ← selection ← calibration
        from .selection import IsoForestAppender, MedianImputeWithIndicator

        # ── MedianImputeWithIndicator ─────────────────────────────────────────

        def _mii_shape(operator):
            n = operator.inputs[0].get_first_dimension()
            f = operator.inputs[0].type.shape[1]
            operator.outputs[0].type = FloatTensorType([n, 2 * f])

        def _mii_convert(scope, operator, container):
            op = operator.raw_operator
            X = operator.inputs[0]
            opv = container.target_opset
            nan = OnnxIsNaN(X, op_version=opv)
            filled = OnnxWhere(
                nan,
                op.medians_.astype(np.float32).reshape(1, -1),
                X,
                op_version=opv,
            )
            ind = OnnxCast(nan, to=_onnx.TensorProto.FLOAT, op_version=opv)
            OnnxConcat(
                filled, ind, axis=1, op_version=opv, output_names=operator.outputs[:1]
            ).add_to(scope, container)

        update_registered_converter(
            MedianImputeWithIndicator,
            "MadMedianImputeWithIndicator",
            _mii_shape,
            _mii_convert,
        )

        # ── IsoForestAppender ─────────────────────────────────────────────────

        def _iso_shape(operator):
            n = operator.inputs[0].get_first_dimension()
            f = operator.inputs[0].type.shape[1]
            operator.outputs[0].type = FloatTensorType([n, f + 1])

        def _iso_convert(scope, operator, container):
            op = operator.raw_operator
            X = operator.inputs[0]
            opv = container.target_opset
            # IsolationForest outputs [label, scores]; index 1 = decision_function
            sub = OnnxSubEstimator(op.iso_, X, op_version=opv)
            score = sub[1]
            OnnxConcat(X, score, axis=1, op_version=opv, output_names=operator.outputs[:1]).add_to(
                scope, container
            )

        update_registered_converter(
            IsoForestAppender,
            "MadIsoForestAppender",
            _iso_shape,
            _iso_convert,
        )

        _CONVERTERS_REGISTERED = True
    except Exception:
        pass  # skl2onnx not available or registration failed; gate will report passed=False


def onnx_export_with_platt(
    pipeline,
    platt: PlattCalibrator,
    n_features: int,
    X_sample: np.ndarray | None = None,
) -> bytes:
    """Export sklearn pipeline + Platt calibration to ONNX bytes.

    The returned model takes a float32 input of shape [N, n_features] and
    outputs a float32 array of shape [N] with calibrated probabilities.

    When platt.use_logit is False (logreg, HGB): exports with raw_scores=True
    so the ONNX graph uses the model's raw margin directly, matching
    decision_function in sklearn.  Avoids float32 saturation from sigmoid→logit.

    When platt.use_logit is True (twostage): keeps the Clip+logit+Platt path.

    Raises ImportError if skl2onnx or onnxruntime are not installed.
    Raises RuntimeError if the pipeline cannot be converted.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    from skl2onnx import to_onnx

    _apply_skl2onnx_shims()
    _register_converters()

    use_logit = platt.use_logit

    if X_sample is None:
        X_sample = np.zeros((5, n_features), dtype=np.float32)

    clf_step = pipeline.steps[-1][1]
    onnx_opts: dict = {"zipmap": False}
    if not use_logit:
        onnx_opts["raw_scores"] = True

    onnx_model = to_onnx(
        pipeline,
        X_sample[:5].astype(np.float32),
        options={id(clf_step): onnx_opts},
        target_opset={"": 17, "ai.onnx.ml": 3},
    )

    # output[1] is probabilities [N,2] (use_logit) or raw class scores [N,2] (not use_logit).
    # For binary classification both shapes are [N, 2]; Gather(axis=1, index=1) extracts
    # the class-1 value (probability or raw margin).
    score_output_name = onnx_model.graph.output[1].name

    # Remove existing outputs; nodes that produce them remain in the graph
    while onnx_model.graph.output:
        onnx_model.graph.output.pop()

    # Gather class-1 value → shape [N]
    gather_out = "__platt_gather"
    idx_name = "__platt_idx"
    idx_tensor = numpy_helper.from_array(np.array(1, dtype=np.int64), name=idx_name)
    onnx_model.graph.initializer.append(idx_tensor)
    gather_node = helper.make_node(
        "Gather", inputs=[score_output_name, idx_name], outputs=[gather_out], axis=1
    )
    onnx_model.graph.node.append(gather_node)

    a_name, b_name = "__platt_a", "__platt_b"
    a_tensor = numpy_helper.from_array(np.array([platt.a], dtype=np.float32), name=a_name)
    b_tensor = numpy_helper.from_array(np.array([platt.b], dtype=np.float32), name=b_name)
    onnx_model.graph.initializer.extend([a_tensor, b_tensor])

    mul_out = "__platt_mul"
    add_out = "__platt_add"
    final_out = "calibrated_probability"

    if not use_logit:
        # Raw-margin path: sigmoid(a * margin + b).  No logit nodes needed.
        onnx_model.graph.node.append(
            helper.make_node("Mul", inputs=[gather_out, a_name], outputs=[mul_out])
        )
        onnx_model.graph.node.append(
            helper.make_node("Add", inputs=[mul_out, b_name], outputs=[add_out])
        )
        onnx_model.graph.node.append(
            helper.make_node("Sigmoid", inputs=[add_out], outputs=[final_out])
        )
    else:
        # Logit path: sigmoid(a * logit(clip(prob)) + b).
        clip_min_name, clip_max_name = "__platt_clip_min", "__platt_clip_max"
        one_name = "__platt_one"
        clip_min_t = numpy_helper.from_array(np.array(1e-6, dtype=np.float32), name=clip_min_name)
        clip_max_t = numpy_helper.from_array(
            np.array(1.0 - 1e-6, dtype=np.float32), name=clip_max_name
        )
        one_t = numpy_helper.from_array(np.array(1.0, dtype=np.float32), name=one_name)
        onnx_model.graph.initializer.extend([clip_min_t, clip_max_t, one_t])

        clip_out = "__platt_clip"
        log_p_out = "__platt_logp"
        one_minus_out = "__platt_1mp"
        log_1mp_out = "__platt_log1mp"
        logit_out = "__platt_logit"

        onnx_model.graph.node.append(
            helper.make_node(
                "Clip", inputs=[gather_out, clip_min_name, clip_max_name], outputs=[clip_out]
            )
        )
        onnx_model.graph.node.append(
            helper.make_node("Log", inputs=[clip_out], outputs=[log_p_out])
        )
        onnx_model.graph.node.append(
            helper.make_node("Sub", inputs=[one_name, clip_out], outputs=[one_minus_out])
        )
        onnx_model.graph.node.append(
            helper.make_node("Log", inputs=[one_minus_out], outputs=[log_1mp_out])
        )
        onnx_model.graph.node.append(
            helper.make_node("Sub", inputs=[log_p_out, log_1mp_out], outputs=[logit_out])
        )
        onnx_model.graph.node.append(
            helper.make_node("Mul", inputs=[logit_out, a_name], outputs=[mul_out])
        )
        onnx_model.graph.node.append(
            helper.make_node("Add", inputs=[mul_out, b_name], outputs=[add_out])
        )
        onnx_model.graph.node.append(
            helper.make_node("Sigmoid", inputs=[add_out], outputs=[final_out])
        )

    final_output_info = helper.make_tensor_value_info(final_out, TensorProto.FLOAT, [None])
    onnx_model.graph.output.append(final_output_info)

    onnx.checker.check_model(onnx_model)
    return onnx_model.SerializeToString()
