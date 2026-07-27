"""DeepLOB inference backends.

Three interchangeable implementations behind one interface:

``StubPredictor``
    Zero drift, uniform class probabilities. Not a placeholder to be deleted --
    it is the permanent degraded mode. If the model artifact is missing or the
    runtime fails to load, the system must keep quoting as pure
    Avellaneda-Stoikov rather than stop. It is also exactly the ``lambda=0``
    ablation arm, so the baseline runs through the identical code path.

``TorchScriptPredictor``
    ``torch.jit`` archive. Portable, no Python dependency at load time, and the
    format the training pipeline exports by default.

``OnnxPredictor``
    ONNX Runtime. Typically the fastest CPU path -- it does constant folding and
    operator fusion ahead of time, which matters against a 5ms p99 budget.

**Latency discipline.** Threads are pinned to 1. Letting torch spin up a thread
pool for a batch-size-1 forward pass adds scheduling jitter that shows up
directly in p99, and there is no throughput to win: we have one sample.

The expected-return conversion is deliberately explicit. The network emits three
class probabilities (down/flat/up); the quoting engine needs a scalar expected
return. We map them through the label threshold ``alpha`` that *defined* the
classes:

    E[r] ~ (P(up) - P(down)) * alpha_scale

so the signal is in the same units as the labels the model was trained on, and
``drift_lambda`` stays interpretable across retrainings.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol

import numpy as np

from amme_shared.logging import get_logger

log = get_logger("deeplob.predictor")


class Predictor(Protocol):
    version: str

    def predict(self, tensor: np.ndarray) -> tuple[np.ndarray, float]:
        """``(class_probabilities[3], latency_us)``."""

    @property
    def ready(self) -> bool: ...


class StubPredictor:
    """Uniform probabilities, zero drift. The safe degraded mode."""

    version = "stub"

    def __init__(self) -> None:
        self._uniform = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)

    @property
    def ready(self) -> bool:
        return True

    def predict(self, tensor: np.ndarray) -> tuple[np.ndarray, float]:
        return self._uniform, 0.0


class TorchScriptPredictor:
    """TorchScript archive on CPU."""

    def __init__(self, path: str | Path, threads: int = 1) -> None:
        import torch

        torch.set_num_threads(threads)
        torch.set_grad_enabled(False)
        self._torch = torch
        self._path = Path(path)
        self.version = f"torchscript:{self._path.name}"
        self._model = torch.jit.load(str(self._path), map_location="cpu")  # type: ignore[no-untyped-call]
        self._model.eval()
        log.info("torchscript_loaded", path=str(self._path), threads=threads)

    @property
    def ready(self) -> bool:
        return True

    def predict(self, tensor: np.ndarray) -> tuple[np.ndarray, float]:
        torch = self._torch
        start = time.perf_counter_ns()
        # (T, F) -> (1, 1, T, F): batch of one, single channel.
        batch = torch.from_numpy(np.ascontiguousarray(tensor, dtype=np.float32)).unsqueeze(
            0
        ).unsqueeze(0)
        with torch.inference_mode():
            logits = self._model(batch)
            probabilities = torch.softmax(logits, dim=-1).squeeze(0).numpy()
        latency_us = (time.perf_counter_ns() - start) / 1000.0
        return probabilities.astype(np.float64), latency_us


class OnnxPredictor:
    """ONNX Runtime on CPU."""

    def __init__(self, path: str | Path, threads: int = 1) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._path = Path(path)
        self.version = f"onnx:{self._path.name}"
        self._session = ort.InferenceSession(
            str(self._path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        log.info("onnx_loaded", path=str(self._path), threads=threads)

    @property
    def ready(self) -> bool:
        return True

    def predict(self, tensor: np.ndarray) -> tuple[np.ndarray, float]:
        start = time.perf_counter_ns()
        batch = np.ascontiguousarray(tensor, dtype=np.float32)[None, None, :, :]
        logits = self._session.run(None, {self._input_name: batch})[0]
        latency_us = (time.perf_counter_ns() - start) / 1000.0
        return _softmax(logits.reshape(-1)).astype(np.float64), latency_us


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)  # subtract max for numerical stability
    exp = np.exp(shifted)
    normalised: np.ndarray = exp / np.sum(exp)
    return normalised


def probabilities_to_drift(probabilities: np.ndarray, alpha_scale: float) -> tuple[float, float]:
    """``(expected_return, confidence)`` from ``[P(down), P(flat), P(up)]``.

    ``confidence`` is the max class probability, which the quoting engine
    thresholds before acting. Using max-softmax rather than the up/down margin
    means a confidently-flat prediction is treated as high confidence in *no
    drift*, which is the correct reading and stops the engine from acting on a
    signal it does not have.
    """
    if probabilities.shape[-1] != 3:
        raise ValueError(f"expected 3 class probabilities, got {probabilities.shape}")
    p_down, _p_flat, p_up = (float(v) for v in probabilities)
    drift = (p_up - p_down) * alpha_scale
    confidence = float(np.max(probabilities))
    return drift, confidence


def build_predictor(
    backend: str, artifact_path: str, onnx_path: str, threads: int = 1
) -> Predictor:
    """Load the configured backend, falling back to the stub on any failure.

    Falling back rather than raising is intentional: an inference service that
    refuses to start takes the drift signal offline *and* leaves the stream
    unconsumed. Degrading to zero-drift keeps the pipeline whole and is exactly
    the pure-A-S behaviour, which is a known-good state.
    """
    try:
        if backend == "torchscript":
            if Path(artifact_path).exists():
                return TorchScriptPredictor(artifact_path, threads)
            log.warning("model_artifact_missing", path=artifact_path, falling_back="stub")
        elif backend == "onnx":
            if Path(onnx_path).exists():
                return OnnxPredictor(onnx_path, threads)
            log.warning("model_artifact_missing", path=onnx_path, falling_back="stub")
        elif backend != "stub":
            log.warning("unknown_backend", backend=backend, falling_back="stub")
    except Exception as exc:
        log.error("predictor_load_failed", backend=backend, error=f"{type(exc).__name__}: {exc}")
    return StubPredictor()


__all__ = [
    "OnnxPredictor",
    "Predictor",
    "StubPredictor",
    "TorchScriptPredictor",
    "build_predictor",
    "probabilities_to_drift",
]
