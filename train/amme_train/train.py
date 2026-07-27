"""DeepLOB training and export. ``python -m amme_train.train``.

Runs on CPU in minutes on the reference dataset; on a free Colab/Kaggle GPU it
scales to hours of recorded data without changes.

Choices worth defending:

**Class weighting.** LOB labels are imbalanced even after alpha tuning, and an
unweighted cross-entropy on 60% "flat" converges to predicting flat always --
95% accuracy, zero information, and a drift signal that is identically zero.
Weights are inverse-frequency from the *training* split only.

**Early stopping on validation macro-F1, not accuracy.** Accuracy rewards the
majority-class collapse above; macro-F1 does not.

**The reported number is test-set, evaluated once.** Validation is used for
model selection, so quoting validation performance as the result is a form of
overfitting to the split.

**Export is verified, not assumed.** ``export_torchscript`` re-loads the archive
and asserts it reproduces the eager model's logits. A silently-wrong export
degrades to a plausible but incorrect drift signal, which is worse than a
missing model because nothing alerts.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from amme_shared.logging import configure_logging
from amme_train.dataset import (
    CLASS_NAMES,
    Dataset,
    DatasetConfig,
    build_from_simulator,
    load_dataset,
)
from amme_train.model import DeepLOB, count_parameters

log = configure_logging("train", "INFO", json_output=False)


@dataclass(frozen=True, slots=True)
class TrainConfig:
    epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 6
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "cpu"
    lstm_hidden: int = 64
    dropout: float = 0.1
    use_attention: bool = True
    num_workers: int = 0


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def class_weights(y: np.ndarray, n_classes: int = 3) -> Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> float:
    """Unweighted mean per-class F1. No sklearn dependency for one metric."""
    scores = []
    for c in range(n_classes):
        tp = float(np.sum((y_pred == c) & (y_true == c)))
        fp = float(np.sum((y_pred == c) & (y_true != c)))
        fn = float(np.sum((y_pred != c) & (y_true == c)))
        denom = 2 * tp + fp + fn
        scores.append(2 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(scores))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n: int = 3) -> np.ndarray:
    matrix = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred, strict=True):
        matrix[t, p] += 1
    return matrix


def _loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool,
            workers: int = 0) -> DataLoader:
    tensors = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(tensors, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, drop_last=False)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    for xb, yb in loader:
        logits = model(xb.to(device))
        preds.append(logits.argmax(dim=1).cpu().numpy())
        trues.append(yb.numpy())
    return np.concatenate(trues), np.concatenate(preds)


def train_model(dataset: Dataset, config: TrainConfig | None = None) -> tuple[DeepLOB, dict]:
    config = config or TrainConfig()
    set_seed(config.seed)
    device = config.device

    model = DeepLOB(
        levels=dataset.config.levels,
        lookback=dataset.config.lookback,
        lstm_hidden=config.lstm_hidden,
        dropout=config.dropout,
        use_attention=config.use_attention,
    ).to(device)
    log.info("model_built", parameters=count_parameters(model),
             attention=config.use_attention)

    train_loader = _loader(dataset.train.x, dataset.train.y, config.batch_size, True,
                           config.num_workers)
    val_loader = _loader(dataset.val.x, dataset.val.y, config.batch_size, False)
    test_loader = _loader(dataset.test.x, dataset.test.y, config.batch_size, False)

    weights = class_weights(dataset.train.y).to(device)
    log.info("class_weights", weights=[round(float(w), 3) for w in weights])
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", factor=0.5, patience=2
    )

    best_f1 = -1.0
    best_state: dict[str, Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        started = time.perf_counter()
        total_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            optimiser.zero_grad(set_to_none=True)
            loss = criterion(model(xb.to(device)), yb.to(device))
            loss.backward()
            # LSTMs on long sequences produce occasional huge gradients; without
            # clipping a single batch can destroy an otherwise-converged model.
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimiser.step()
            total_loss += float(loss.item())
            n_batches += 1

        y_true, y_pred = evaluate(model, val_loader, device)
        val_f1 = macro_f1(y_true, y_pred)
        val_acc = float(np.mean(y_true == y_pred))
        scheduler.step(val_f1)

        history.append({
            "epoch": epoch,
            "train_loss": total_loss / max(1, n_batches),
            "val_macro_f1": val_f1,
            "val_accuracy": val_acc,
            "seconds": time.perf_counter() - started,
        })
        log.info("epoch", epoch=epoch, loss=round(total_loss / max(1, n_batches), 4),
                 val_f1=round(val_f1, 4), val_acc=round(val_acc, 4),
                 secs=round(time.perf_counter() - started, 1))

        if val_f1 > best_f1 + 1e-5:
            best_f1 = val_f1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                log.info("early_stop", epoch=epoch, best_val_f1=round(best_f1, 4))
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test set touched exactly once, after selection is finished.
    y_true, y_pred = evaluate(model, test_loader, device)
    metrics = {
        "val_macro_f1": best_f1,
        "test_macro_f1": macro_f1(y_true, y_pred),
        "test_accuracy": float(np.mean(y_true == y_pred)),
        "test_confusion": confusion_matrix(y_true, y_pred).tolist(),
        "test_class_balance": dataset.test.class_balance(),
        "majority_class_accuracy": max(dataset.test.class_balance().values()),
        "n_parameters": count_parameters(model),
        "history": history,
    }
    # The comparison that decides whether the model is worth deploying at all.
    metrics["beats_majority_baseline"] = bool(
        metrics["test_accuracy"] > metrics["majority_class_accuracy"]
    )
    log.info("test_metrics", macro_f1=round(metrics["test_macro_f1"], 4),
             accuracy=round(metrics["test_accuracy"], 4),
             majority_baseline=round(metrics["majority_class_accuracy"], 4),
             beats_baseline=metrics["beats_majority_baseline"])
    return model, metrics


def export_torchscript(model: DeepLOB, path: str | Path, lookback: int,
                       levels: int) -> Path:
    """Trace to TorchScript and verify the archive reproduces the eager model."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    example = torch.randn(1, 1, lookback, levels * 4)

    with torch.no_grad():
        scripted = torch.jit.trace(model, example)
        scripted = torch.jit.freeze(scripted)   # inline params, drop training branches
        torch.jit.save(scripted, str(target))

        reloaded = torch.jit.load(str(target))
        expected = model(example)
        actual = reloaded(example)
    if not torch.allclose(expected, actual, atol=1e-5):
        raise RuntimeError(
            f"TorchScript export does not match the eager model "
            f"(max diff {float((expected - actual).abs().max()):.2e})"
        )
    log.info("torchscript_exported", path=str(target),
             size_kb=round(target.stat().st_size / 1024, 1))
    return target


def export_onnx(model: DeepLOB, path: str | Path, lookback: int, levels: int) -> Path | None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    example = torch.randn(1, 1, lookback, levels * 4)
    try:
        torch.onnx.export(
            model, example, str(target),
            input_names=["lob"], output_names=["logits"],
            dynamic_axes={"lob": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
        )
    except Exception as exc:
        log.warning("onnx_export_failed", error=f"{type(exc).__name__}: {exc}")
        return None
    log.info("onnx_exported", path=str(target))
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train DeepLOB and export artifacts")
    parser.add_argument("--dataset", type=str, default=None, help="path to a saved .npz")
    parser.add_argument("--seconds", type=float, default=7200.0,
                        help="simulator seconds when no dataset is given")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128, dest="batch_size")
    parser.add_argument("--lookback", type=int, default=100)
    parser.add_argument("--levels", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--out", type=str, default="models/deeplob.ts")
    parser.add_argument("--onnx", type=str, default="models/deeplob.onnx")
    parser.add_argument("--metrics", type=str, default="artifacts/deeplob_metrics.json")
    parser.add_argument("--no-attention", action="store_false", dest="attention")
    args = parser.parse_args(argv)

    ds_config = DatasetConfig(levels=args.levels, lookback=args.lookback,
                              horizon=args.horizon)
    if args.dataset:
        dataset = load_dataset(args.dataset, ds_config)
    else:
        log.info("building_dataset", seconds=args.seconds, seed=args.seed)
        dataset = build_from_simulator(args.seconds, args.seed, ds_config)
    print("dataset splits (chronological, purged):")
    print(dataset.summary())

    model, metrics = train_model(
        dataset,
        TrainConfig(epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
                    use_attention=args.attention),
    )

    export_torchscript(model, args.out, args.lookback, args.levels)
    export_onnx(model, args.onnx, args.lookback, args.levels)

    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {"metrics": metrics, "dataset": asdict(ds_config),
             "splits": {s.name: {"n": len(s), **s.class_balance()}
                        for s in (dataset.train, dataset.val, dataset.test)}},
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== test set (evaluated once) ===")
    print(f"  macro F1          {metrics['test_macro_f1']:.4f}")
    print(f"  accuracy          {metrics['test_accuracy']:.4f}")
    print(f"  majority baseline {metrics['majority_class_accuracy']:.4f}")
    print(f"  beats baseline    {metrics['beats_majority_baseline']}")
    print("  confusion (rows=true, cols=pred), classes " + str(CLASS_NAMES))
    for row in metrics["test_confusion"]:
        print("    " + "  ".join(f"{v:>7,}" for v in row))
    return 0 if metrics["beats_majority_baseline"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
