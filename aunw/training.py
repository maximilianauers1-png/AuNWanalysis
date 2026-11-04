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

The `TrainingMonitor` records the run in <model_dir>/progress.json. The GUI
starts training as a separate process (`start_process`) and follows that file,
so the interface stays responsive and the run survives page changes or a
closed browser. TensorBoard logs are written as well (`tensorboard --logdir models`).
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import keras
import numpy as np
import pandas as pd

from .config import ROOT, ModelConfig, TrainConfig, save_json
from .dataset import load_split, make_dataset, mixed_dataset
from .model import build_unet, compile_model, enable_mixed_precision, load_model


class TrainingMonitor:
    """Progress record of one training run.

    It is persisted to <model_dir>/progress.json (+ preview.npz) so that any
    other process - the GUI - can follow the run. Training is therefore run as
    a separate process (see `start_process`): a crash or a long computation in
    the trainer can never freeze or kill the user interface, and the run
    continues even if the browser is closed. A stop is requested by creating
    the file <model_dir>/STOP.
    """

    FIELDS = ("status", "message", "epoch", "total_epochs", "step", "steps_per_epoch",
              "history", "batch_loss", "started")

    def __init__(self, model_dir: Path | str | None = None):
        self.model_dir = Path(model_dir) if model_dir else None
        self.status, self.message = "idle", ""                 # idle|running|finished|stopped|error
        self.epoch = self.total_epochs = self.step = self.steps_per_epoch = 0
        self.history: list[dict] = []                           # one dict per epoch
        self.batch_loss: list[float] = []
        self.started = 0.0
        self.preview: dict = {}
        self._last_save = 0.0

    # ---- writing (trainer side)
    def update(self, force: bool = False, **values) -> None:
        for k, v in values.items():
            setattr(self, k, v)
        if self.model_dir and (force or time.time() - self._last_save > 1.0):
            self._last_save = time.time()
            data = {k: getattr(self, k) for k in self.FIELDS}
            data["batch_loss"] = data["batch_loss"][-2000:]
            tmp = self.model_dir / "progress.json.tmp"
            tmp.write_text(json.dumps(data))
            tmp.replace(self.model_dir / "progress.json")      # atomic: readers never see half a file

    def set_preview(self, preview: dict) -> None:
        self.preview = preview
        if self.model_dir:
            with open(self.model_dir / "preview.npz.tmp", "wb") as f:
                np.savez_compressed(f, **preview)
            (self.model_dir / "preview.npz.tmp").replace(self.model_dir / "preview.npz")

    def stop_requested(self) -> bool:
        return bool(self.model_dir and (self.model_dir / "STOP").exists())

    # ---- reading (GUI side)
    @staticmethod
    def load(model_dir: Path | str) -> dict | None:
        model_dir = Path(model_dir)
        try:
            data = json.loads((model_dir / "progress.json").read_text())
        except (OSError, json.JSONDecodeError):
            return None
        data["elapsed"] = time.time() - data["started"] if data.get("started") else 0.0
        data["preview"] = {}
        if (model_dir / "preview.npz").exists():
            try:
                with np.load(model_dir / "preview.npz") as f:
                    data["preview"] = {k: f[k] for k in f.files}
            except (OSError, ValueError):
                pass
        return data


class MonitorCallback(keras.callbacks.Callback):
    """Pushes progress into a TrainingMonitor, prints one line per epoch, honours STOP."""

    def __init__(self, monitor: TrainingMonitor, preview_x: np.ndarray, preview_y: dict):
        super().__init__()
        self.mon, self.px, self.py = monitor, preview_x, preview_y
        self.t0 = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.time()
        self.mon.update(epoch=epoch + 1, step=0)

    def on_train_batch_end(self, batch, logs=None):
        self.mon.batch_loss.append(float(logs["loss"]))
        self.mon.update(step=batch + 1)
        if self.mon.stop_requested():
            self.model.stop_training = True

    def on_epoch_end(self, epoch, logs=None):
        lr = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
        row = {"epoch": epoch + 1, "lr": lr, "seconds": time.time() - self.t0,
               **{k: float(v) for k, v in (logs or {}).items() if k != "learning_rate"}}
        out = self.model.predict_on_batch(self.px)
        self.mon.set_preview({"image": self.px[..., 0], "mask_true": self.py["mask"][..., 0],
                              "mask_pred": np.asarray(out["mask"])[..., 0],
                              "centerline_true": self.py["centerline"][..., 0],
                              "centerline_pred": np.asarray(out["centerline"])[..., 0],
                              "epoch": np.array(epoch + 1)})
        self.mon.history.append(row)
        self.mon.update(force=True)
        print(f"epoch {row['epoch']:3d}  loss {row['loss']:.4f}  val_loss {row.get('val_loss', np.nan):.4f}  "
              f"mask dice {row.get('val_mask_dice_coef', np.nan):.3f}  "
              f"cl dice {row.get('val_centerline_dice_coef', np.nan):.3f}  "
              f"angle err {row.get('val_orientation_angle_error_deg', np.nan):.1f} deg  "
              f"lr {lr:.1e}  {row['seconds']:.0f}s", flush=True)


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
        monitor.update(force=True, status="stopped" if monitor.stop_requested() else "finished",
                       message=f"Model saved to {model_dir}")


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
        monitor.update(force=True, status="running", started=time.time(),
                       total_epochs=train_cfg.epochs,
                       steps_per_epoch=len(train_arr["image"]) // train_cfg.batch_size)
    model.fit(train_ds, validation_data=val_ds, epochs=train_cfg.epochs, verbose=0,
              callbacks=_callbacks(model_dir, train_cfg, monitor, val_arr))
    _finish(model, model_dir, monitor, model_config=model_cfg, train=train_cfg, data_dir=str(data_dir),
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
        monitor.update(force=True, status="running", started=time.time(),
                       total_epochs=train_cfg.epochs, steps_per_epoch=steps)
    model.fit(train_ds, validation_data=val_ds, epochs=train_cfg.epochs, steps_per_epoch=steps,
              verbose=0, callbacks=_callbacks(model_dir, train_cfg, monitor, real_val))
    _finish(model, model_dir, monitor, train=train_cfg, base_model=str(base_model),
            kind="finetuned", freeze_encoder=freeze_encoder, real_fraction=real_fraction)
    return model


def start_process(args: list[str], model_dir: Path) -> subprocess.Popen:
    """Launch `python -m aunw.cli <args>` as a separate process, logging to model_dir/train.log.

    This is exactly the command a user could type in a terminal; the GUI only
    starts it and then follows progress.json.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "STOP").unlink(missing_ok=True)
    TrainingMonitor(model_dir).update(force=True, status="starting", message="starting …")
    log = open(model_dir / "train.log", "w")
    return subprocess.Popen([sys.executable, "-m", "aunw.cli", *args], stdout=log,
                            stderr=subprocess.STDOUT, cwd=str(ROOT))


def request_stop(model_dir: Path) -> None:
    """Ask a running trainer to stop after the current batch (best weights are kept)."""
    (Path(model_dir) / "STOP").touch()


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

