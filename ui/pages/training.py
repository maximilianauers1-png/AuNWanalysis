"""Step 3: train the network and watch it learn (live).

Training runs as a separate process (the same command as on the terminal);
this page only starts it and follows <model_dir>/progress.json.
"""
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from aunw.config import MODELS_DIR, TrainConfig
from aunw.plots import prediction_preview, training_curves
from aunw.training import (TrainingMonitor, model_file, overfitting_report, request_stop,
                           start_process)
from ui import common


def _settings() -> tuple[TrainConfig, str, str | None]:
    cfg: TrainConfig = st.session_state["train_cfg"]
    c = st.columns(4)
    cfg.epochs = c[0].number_input("Max. epochs", 1, 500, cfg.epochs)
    cfg.batch_size = c[1].selectbox("Batch size", [2, 4, 8, 16, 32], index=[2, 4, 8, 16, 32].index(cfg.batch_size))
    cfg.learning_rate = c[2].select_slider("Learning rate", [1e-4, 3e-4, 1e-3, 3e-3], cfg.learning_rate,
                                           format_func=lambda v: f"{v:.0e}")
    cfg.weight_decay = c[3].select_slider("Weight decay (AdamW)", [0.0, 1e-5, 1e-4, 1e-3],
                                          cfg.weight_decay, format_func=lambda v: f"{v:.0e}")
    c = st.columns(4)
    cfg.early_stop_patience = c[0].number_input("Early-stopping patience", 2, 100, cfg.early_stop_patience,
                                                help="Stop after this many epochs without "
                                                     "improvement of the validation loss.")
    cfg.augment = c[2].toggle("Data augmentation", cfg.augment)
    cfg.mixed_precision = c[3].toggle("Mixed precision (GPU)", cfg.mixed_precision)
    st.session_state["data_dir"] = st.text_input(
        "Dataset folder (from step 1)", st.session_state["data_dir"],
        help="Folder containing train.npz and val.npz.")
    c = st.columns([2, 2])
    mcfg = st.session_state["model_cfg"]
    name = c[0].text_input("Name of the new model",
                           f"unet_f{mcfg.base_filters}_d{mcfg.depth}_{datetime.now():%Y%m%d_%H%M}")
    init = None
    if common.available_models():
        if c[1].toggle("Continue from the active network", False):
            init = str(model_file(Path(st.session_state["model_dir"])))
    return cfg, name, init


def is_running() -> bool:
    proc = common.trainer_slot()["process"]
    return proc is not None and proc.poll() is None


def launch(args: list[str], model_dir: Path) -> None:
    slot = common.trainer_slot()
    slot["process"] = start_process(args, model_dir)
    slot["model_dir"] = str(model_dir)
    slot["command"] = "python -m aunw.cli " + " ".join(args)


@st.fragment(run_every=3)
def monitor_panel() -> None:
    slot = common.trainer_slot()
    if not slot["model_dir"]:
        st.info("No training run in this session yet.")
        return
    s = TrainingMonitor.load(slot["model_dir"])
    proc = slot["process"]
    crashed = proc is not None and proc.poll() not in (None, 0)
    if s is None or s["status"] == "error" or (crashed and s["status"] in ("starting", "running")):
        log = Path(slot["model_dir"]) / "train.log"
        tail = log.read_text()[-3000:] if log.exists() else ""
        if crashed or (s and s["status"] == "error"):
            st.error("Training stopped with an error. Last lines of the log:")
            st.code(tail or (s or {}).get("message", ""))
        else:
            st.info("Starting the training process …")
        return
    if s["status"] == "starting":
        st.info("Starting the training process (loading data, building the network) …")
        return
    h = pd.DataFrame(s["history"])
    rep = overfitting_report(h)

    frac_epoch = s["step"] / max(s["steps_per_epoch"], 1)
    st.progress(min(1.0, (max(s["epoch"] - 1, 0) + frac_epoch) / max(s["total_epochs"], 1)),
                f"{s['status'].upper()} · epoch {s['epoch']}/{s['total_epochs']} · "
                f"step {s['step']}/{s['steps_per_epoch']} · {s['elapsed'] / 60:.1f} min")
    st.caption(f"Command: `{slot.get('command', '')}` · log: `{slot['model_dir']}/train.log`")
    if s["status"] in ("finished", "stopped"):
        st.success(s["message"] + (" (stopped by user; best epoch kept)" if s["status"] == "stopped" else ""))

    m = st.columns(5)
    if len(h):
        last = h.iloc[-1]
        m[0].metric("Validation loss", f"{last['val_loss']:.4f}",
                    f"{last['val_loss'] - h['val_loss'].iloc[-2]:+.4f}" if len(h) > 1 else None,
                    delta_color="inverse")
        m[1].metric("Mask Dice (val)", f"{last.get('val_mask_dice_coef', float('nan')):.3f}")
        m[2].metric("Centre-line Dice (val)", f"{last.get('val_centerline_dice_coef', float('nan')):.3f}")
        m[3].metric("Angle error (val)", f"{last.get('val_orientation_angle_error_deg', float('nan')):.1f}°")
        m[4].metric("Best epoch", rep["best_epoch"], f"{rep['epochs_since_best']} epochs ago",
                    delta_color="off")
        if rep["overfitting"]:
            st.warning("⚠️ **Possible over-fitting**: validation loss has risen over the last epochs "
                       "while training loss kept falling. Early stopping will restore the best "
                       "weights automatically.")
        else:
            st.caption(f"Generalisation gap (val − train loss): {rep['gap']:+.4f} · "
                       f"ratio {rep['ratio']:.2f}. A growing positive gap indicates over-fitting.")
        st.pyplot(training_curves(h), width="stretch")
    if s["batch_loss"]:
        st.caption("Loss of every training batch (noisy by nature; the trend matters)")
        st.line_chart(pd.DataFrame({"batch loss": s["batch_loss"]}), height=160)
    if s["preview"]:
        st.markdown(f"**Validation examples after epoch {int(s['preview']['epoch'])}**")
        st.pyplot(prediction_preview(s["preview"]), width="stretch")


def render() -> None:
    common.header(3, "Training", "Teach the network on the synthetic dataset and monitor it live.")
    running = is_running()
    if not common.dataset_ready():
        st.warning("No dataset found. Generate one in step 1 or enter its folder below.")
    with st.expander("Settings", expanded=not running):
        cfg, name, init = _settings()
        with st.popover("ℹ️ How over-fitting is prevented"):
            st.markdown(
                "- independent validation set (different random seed)\n"
                "- **early stopping** with restoration of the best weights\n"
                "- **learning-rate reduction** on plateaus\n"
                "- **AdamW** decoupled weight decay, **SpatialDropout**, BatchNorm\n"
                "- random flips/rotations and contrast/noise augmentation")

    c = st.columns([1, 1, 3])
    if c[0].button("▶ Start training", type="primary", disabled=running or not common.dataset_ready()):
        mcfg = st.session_state["model_cfg"]
        args = ["train", "--data", st.session_state["data_dir"], "--out", str(MODELS_DIR / name),
                "--epochs", str(cfg.epochs), "--batch-size", str(cfg.batch_size),
                "--lr", str(cfg.learning_rate), "--weight-decay", str(cfg.weight_decay),
                "--patience", str(cfg.early_stop_patience), "--base-filters", str(mcfg.base_filters),
                "--depth", str(mcfg.depth), "--dropout", str(mcfg.dropout)]
        args += ["--init", init] if init else []
        args += [] if cfg.augment else ["--no-augment"]
        args += ["--mixed-precision"] if cfg.mixed_precision else []
        launch(args, MODELS_DIR / name)
        st.session_state["model_dir"] = str(MODELS_DIR / name)
        st.rerun()
    if c[1].button("■ Stop", disabled=not running):
        request_stop(Path(common.trainer_slot()["model_dir"]))
        st.toast("Stopping after the current batch – the best epoch is kept …")
    c[2].caption("Training runs as a separate process: you can switch steps or even close the "
                 "browser meanwhile. Detailed logs: `tensorboard --logdir models`.")
    monitor_panel()
