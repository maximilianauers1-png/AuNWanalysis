"""Training and fine-tuning with live monitoring.

Over-fitting countermeasures used here (and why)
    * Independent validation set (different generator seed / different real
      images): the only honest estimate of generalisation.
    * EarlyStopping(restore_best_weights=True): stop when the validation loss
      has not improved for `patience` epochs and roll back to the best epoch.
    * ReduceLROnPlateau: halve the learning rate when progress stalls, which
      often resolves an apparent plateau before early stopping triggers.
    * AdamW weight decay, SpatialDropout2D, BatchNorm (see model.py) and data
      augmentation (see dataset.py).
    * Fine-tuning on real data: encoder frozen, small learning rate and real
      tiles mixed with synthetic ones, so a handful of annotated images cannot
      be memorised and old knowledge is not forgotten.

The `TrainingMonitor` is a thread-safe record of the training run. The GUI
reads it while training runs in a background thread, so the interface stays
responsive and the run survives page changes. TensorBoard logs are written as
well (`tensorboard --logdir models`).
"""
from __future__ import annotations

import importlib.util
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import keras
import numpy as np
import pandas as pd

from .config import ModelConfig, TrainConfig, save_json
from .dataset import load_split, make_dataset, mixed_dataset
from .model import OUTPUTS, build_unet, compile_model, enable_mixed_precision, load_model


@dataclass
class TrainingMonitor:
    status: str = "idle"                 # idle | running | finished | stopped | error
    message: str = ""
    epoch: int = 0
    total_epochs: int = 0
    step: int = 0
    steps_per_epoch: int = 0
    history: list[dict] = field(default_factory=list)     # one dict per epoch
    batch_loss: list[float] = field(default_factory=list)
    preview: dict = field(default_factory=dict)            # latest validation predictions
    started: float = 0.0
    model_dir: str = ""
    stop_event: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> dict:
        with self.lock:
            return {"status": self.status, "message": self.message, "epoch": self.epoch,
                    "total_epochs": self.total_epochs, "step": self.step,
                    "steps_per_epoch": self.steps_per_epoch, "history": list(self.history),
                    "batch_loss": list(self.batch_loss[-2000:]), "preview": dict(self.preview),
                    "elapsed": time.time() - self.started if self.started else 0.0,
                    "model_dir": self.model_dir}


class MonitorCallback(keras.callbacks.Callback):
    """Pushes progress into a TrainingMonitor and honours stop requests."""

    def __init__(self, monitor: TrainingMonitor, preview_x: np.ndarray, preview_y: dict,
                 preview_every: int = 1):
        super().__init__()
        self.mon, self.px, self.py, self.every = monitor, preview_x, preview_y, preview_every
        self.t0 = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.time()
        with self.mon.lock:
            self.mon.epoch, self.mon.step = epoch + 1, 0

    def on_train_batch_end(self, batch, logs=None):
        with self.mon.lock:
            self.mon.step = batch + 1
            self.mon.batch_loss.append(float(logs["loss"]))
        if self.mon.stop_event.is_set():
            self.model.stop_training = True

    def on_epoch_end(self, epoch, logs=None):
        lr = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
        row = {"epoch": epoch + 1, "lr": lr, "seconds": time.time() - self.t0,
               **{k: float(v) for k, v in (logs or {}).items() if k != "learning_rate"}}
        preview = {}
        if (epoch + 1) % self.every == 0:
            out = self.model.predict_on_batch(self.px)
            preview = {"image": self.px[..., 0], "mask_true": self.py["mask"][..., 0],
                       "mask_pred": np.asarray(out["mask"])[..., 0],
                       "centerline_true": self.py["centerline"][..., 0],
                       "centerline_pred": np.asarray(out["centerline"])[..., 0],
                       "epoch": epoch + 1}
        with self.mon.lock:
            self.mon.history.append(row)
            if preview:
                self.mon.preview = preview


def _preview_batch(val: dict, n: int = 3):
    ds = make_dataset({k: v[:n] for k, v in val.items()}, n, training=False)
    x, y = next(iter(ds))
    return np.asarray(x), {k: np.asarray(v) for k, v in y.items()}


def _callbacks(model_dir: Path, cfg: TrainConfig, monitor: TrainingMonitor | None, val: dict):
    cbs = [
        keras.callbacks.ModelCheckpoint(str(model_dir / "best.keras"), monitor="val_loss",
                                        save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg.early_stop_patience,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=cfg.lr_patience, min_lr=1e-6),
        keras.callbacks.CSVLogger(str(model_dir / "history.csv")),
    ]
    if importlib.util.find_spec("tensorboard"):           # optional dependency
        cbs.append(keras.callbacks.TensorBoard(log_dir=str(model_dir / "logs")))
    if monitor is not None:
        cbs.append(MonitorCallback(monitor, *_preview_batch(val)))
    return cbs


def _finish(model: keras.Model, model_dir: Path, monitor: TrainingMonitor | None, **configs):
    model.save(model_dir / "model.keras")
    save_json(model_dir / "config.json", **configs)
    if monitor is not None:
        with monitor.lock:
            monitor.status = "stopped" if monitor.stop_event.is_set() else "finished"
            monitor.message = f"Model saved to {model_dir}"


# --------------------------------------------------------------------------- training
def train(model_dir: Path, data_dir: Path, model_cfg: ModelConfig, train_cfg: TrainConfig,
          monitor: TrainingMonitor | None = None, init_from: Path | None = None) -> keras.Model:
    """Train on a synthetic dataset (train.npz / val.npz in `data_dir`)."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(train_cfg.seed)
    enable_mixed_precision(train_cfg.mixed_precision)

    train_arr, val_arr = load_split(Path(data_dir) / "train.npz"), load_split(Path(data_dir) / "val.npz")
    train_ds = make_dataset(train_arr, train_cfg.batch_size, True, train_cfg.augment, train_cfg.seed)
    val_ds = make_dataset(val_arr, train_cfg.batch_size, False)

    model = load_model(init_from) if init_from else build_unet(model_cfg)
    compile_model(model, train_cfg)
    if monitor is not None:
        with monitor.lock:
            monitor.status, monitor.started = "running", time.time()
            monitor.total_epochs = train_cfg.epochs
            monitor.steps_per_epoch = len(train_arr["image"]) // train_cfg.batch_size
            monitor.model_dir = str(model_dir)
    model.fit(train_ds, validation_data=val_ds, epochs=train_cfg.epochs, verbose=0,
              callbacks=_callbacks(model_dir, train_cfg, monitor, val_arr))
    _finish(model, model_dir, monitor, model=model_cfg, train=train_cfg, data_dir=str(data_dir),
            kind="synthetic")
    return model


def finetune(base_model: Path, model_dir: Path, real_train: dict, real_val: dict,
             synthetic: dict, train_cfg: TrainConfig, monitor: TrainingMonitor | None = None,
             freeze_encoder: bool = True, real_fraction: float = 0.5) -> keras.Model:
    """Adapt a synthetic-trained model to real SEM images.

    real_train / real_val must come from *different images* (split by image,
    not by tile), otherwise neighbouring tiles leak information and the
    validation loss is over-optimistic.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(train_cfg.seed)
    model = load_model(base_model)
    if freeze_encoder:
        for layer in model.layers:
            if layer.name.startswith(("enc", "pool")):
                layer.trainable = False                     # BN layers then also run in inference mode
    compile_model(model, train_cfg)

    steps = max(20, 2 * len(real_train["image"]) // train_cfg.batch_size)
    train_ds = mixed_dataset(real_train, synthetic, train_cfg.batch_size, real_fraction, train_cfg.seed)
    val_ds = make_dataset(real_val, train_cfg.batch_size, False)
    if monitor is not None:
        with monitor.lock:
            monitor.status, monitor.started = "running", time.time()
            monitor.total_epochs, monitor.steps_per_epoch = train_cfg.epochs, steps
            monitor.model_dir = str(model_dir)
    model.fit(train_ds, validation_data=val_ds, epochs=train_cfg.epochs, steps_per_epoch=steps,
              verbose=0, callbacks=_callbacks(model_dir, train_cfg, monitor, real_val))
    _finish(model, model_dir, monitor, train=train_cfg, base_model=str(base_model),
            kind="finetuned", freeze_encoder=freeze_encoder, real_fraction=real_fraction)
    return model


def run_in_background(fn, monitor: TrainingMonitor, **kwargs) -> threading.Thread:
    """Run train()/finetune() in a daemon thread; errors are reported via the monitor."""
    def target():
        try:
            fn(monitor=monitor, **kwargs)
        except Exception as exc:                             # surfaced in the GUI
            with monitor.lock:
                monitor.status = "error"
                monitor.message = f"{exc}\n{traceback.format_exc()}"

    thread = threading.Thread(target=target, daemon=True, name="aunw-training")
    thread.start()
    return thread


# --------------------------------------------------------------------------- diagnostics
def overfitting_report(history: list[dict] | pd.DataFrame, window: int = 3) -> dict:
    """Simple, interpretable over-fitting diagnostics from the loss curves.

    generalisation gap  g_t = L_val(t) - L_train(t)
    over-fitting flag   validation loss rose over the last `window` epochs while
                        the training loss kept falling.
    """
    h = pd.DataFrame(history)
    if len(h) < 2 or "val_loss" not in h:
        return {"gap": np.nan, "ratio": np.nan, "best_epoch": None, "overfitting": False,
                "epochs_since_best": 0}
    gap = float(h["val_loss"].iloc[-1] - h["loss"].iloc[-1])
    best = int(h["val_loss"].idxmin())
    flag = False
    if len(h) > window:
        dv = h["val_loss"].iloc[-1] - h["val_loss"].iloc[-1 - window]
        dt = h["loss"].iloc[-1] - h["loss"].iloc[-1 - window]
        flag = bool(dv > 0 and dt < 0)
    return {"gap": gap, "ratio": float(h["val_loss"].iloc[-1] / h["loss"].iloc[-1]),
            "best_epoch": int(h["epoch"].iloc[best]), "overfitting": flag,
            "epochs_since_best": int(len(h) - 1 - best)}


def list_models(models_dir: Path) -> list[Path]:
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return []
    found = [p for p in models_dir.iterdir() if (p / "model.keras").exists() or (p / "best.keras").exists()]
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def model_file(model_dir: Path) -> Path:
    model_dir = Path(model_dir)
    return model_dir / "model.keras" if (model_dir / "model.keras").exists() else model_dir / "best.keras"


__all__ = ["TrainingMonitor", "train", "finetune", "run_in_background", "overfitting_report",
           "list_models", "model_file", "OUTPUTS"]
