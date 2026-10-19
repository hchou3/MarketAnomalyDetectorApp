"""Platt (sigmoid) calibration with explicit (a, b) parameters.

PlattCalibrator fits by minimising binary cross-entropy via L-BFGS-B.
onnx_export_with_platt() appends the sigmoid node to an existing ONNX model
so that (pipeline + Platt) can be verified end-to-end.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator


class PlattCalibrator(BaseEstimator):
    """Sigmoid calibration: P(y=1|score) = sigmoid(a * score + b).

    Fitted by minimising binary cross-entropy (NLL) on (scores, labels).
    Two parameters make this suitable for windows with >=8 positives;
    contrast isotonic regression which overfits small windows.
    """

    def __init__(self) -> None:
        self.a: float | None = None
        self.b: float | None = None

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> PlattCalibrator:
        ys = np.asarray(scores, dtype=float)
        yt = np.asarray(labels, dtype=float)

        def nll(params: np.ndarray) -> float:
            a, b = params
            p = np.clip(expit(a * ys + b), 1e-15, 1 - 1e-15)
            return -float(np.mean(yt * np.log(p) + (1 - yt) * np.log(1 - p)))

        res = minimize(nll, [1.0, 0.0], method="L-BFGS-B")
        self.a, self.b = float(res.x[0]), float(res.x[1])
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if self.a is None:
            raise RuntimeError("Call fit() before transform().")
        return expit(self.a * np.asarray(scores, dtype=float) + self.b)

    def fit_transform(self, scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return self.fit(scores, labels).transform(scores)


def onnx_export_with_platt(pipeline, platt: PlattCalibrator, n_features: int) -> bytes:
    """Export sklearn pipeline + Platt calibration to ONNX bytes.

    The returned model takes a float32 input of shape [N, n_features] and
    outputs a float32 array of shape [N] with calibrated probabilities.

    Raises ImportError if skl2onnx or onnxruntime are not installed.
    Raises RuntimeError if the pipeline cannot be converted.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    # zipmap=False ensures probabilities come out as a float32 tensor, not a ZipMap sequence
    onnx_model = convert_sklearn(
        pipeline,
        initial_types=initial_type,
        target_opset=18,
        options={"zipmap": False},
    )

    # Find the probability output (shape [N, 2]) — skl2onnx names it "probabilities"
    prob_output_name = None
    for output in onnx_model.graph.output:
        if "probab" in output.name.lower():
            prob_output_name = output.name
            break
    if prob_output_name is None:
        prob_output_name = onnx_model.graph.output[-1].name

    # Remove existing outputs so we can add our own
    while onnx_model.graph.output:
        onnx_model.graph.output.pop()

    # Gather index 1 along axis 1 to get class-1 probs (shape [N])
    gather_out = "__platt_gather"
    idx_name = "__platt_idx"
    idx_tensor = numpy_helper.from_array(np.array([1], dtype=np.int64), name=idx_name)
    onnx_model.graph.initializer.append(idx_tensor)
    gather_node = helper.make_node(
        "Gather", inputs=[prob_output_name, idx_name], outputs=[gather_out], axis=1
    )
    onnx_model.graph.node.append(gather_node)

    # Platt sigmoid: sigmoid(a * prob1 + b)
    a_name, b_name = "__platt_a", "__platt_b"
    a_tensor = numpy_helper.from_array(np.array([platt.a], dtype=np.float32), name=a_name)
    b_tensor = numpy_helper.from_array(np.array([platt.b], dtype=np.float32), name=b_name)
    onnx_model.graph.initializer.extend([a_tensor, b_tensor])

    mul_out = "__platt_mul"
    add_out = "__platt_add"
    final_out = "calibrated_probability"

    mul_node = helper.make_node("Mul", inputs=[gather_out, a_name], outputs=[mul_out])
    add_node = helper.make_node("Add", inputs=[mul_out, b_name], outputs=[add_out])
    sig_node = helper.make_node("Sigmoid", inputs=[add_out], outputs=[final_out])
    onnx_model.graph.node.extend([mul_node, add_node, sig_node])

    final_output_info = helper.make_tensor_value_info(final_out, TensorProto.FLOAT, [None])
    onnx_model.graph.output.append(final_output_info)

    onnx.checker.check_model(onnx_model)
    return onnx_model.SerializeToString()
