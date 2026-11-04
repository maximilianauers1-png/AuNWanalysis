"""Step 3: train the network and watch it learn (live)."""
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from aunw.config import MODELS_DIR, TrainConfig
from aunw.plots import prediction_preview, training_curves
from aunw.training import TrainingMonitor, overfitting_report, run_in_background, train, model_file
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
    cfg.lr_patience = c[1].number_input("LR-halving patience", 1, 50, cfg.lr_patience)
    cfg.augment = c[2].toggle("Data augmentation", cfg.augment)
    cfg.mixed_precision = c[3].toggle("Mixed precision (GPU)", cfg.mixed_precision)
    c = st.columns([2, 2])
    mcfg = st.session_state["model_cfg"]
    name = c[0].text_input("Name of the new model",
                           f"unet_f{mcfg.base_filters}_d{mcfg.depth}_{datetime.now():%Y%m%d_%H%M}")
    init = None
    if common.available_models():
        if c[1].toggle("Continue from the active network", False):
            init = str(model_file(Path(st.session_state["model_dir"])))
    return cfg, name, init


@st.fragment(run_every=3)
def monitor_panel() -> None:
    mon: TrainingMonitor | None = common.trainer_slot()["monitor"]
    if mon is None:
        st.info("No training run in this session yet.")
        return
    s = mon.snapshot()
    if s["status"] == "error":
        st.error("Training failed:\n\n" + s["message"])
        return
    h = pd.DataFrame(s["history"])
    rep = overfitting_report(h)

    frac_epoch = s["step"] / max(s["steps_per_epoch"], 1)
    st.progress(min(1.0, (max(s["epoch"] - 1, 0) + frac_epoch) / max(s["total_epochs"], 1)),
                f"{s['status'].upper()} · epoch {s['epoch']}/{s['total_epochs']} · "
                f"step {s['step']}/{s['steps_per_epoch']} · {s['elapsed'] / 60:.1f} min")
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
        st.markdown(f"**Validation examples after epoch {s['preview']['epoch']}**")
        st.pyplot(prediction_preview(s["preview"]), width="stretch")


def render() -> None:
    common.header(3, "Training", "Teach the network on the synthetic dataset and monitor it live.")
    slot = common.trainer_slot()
    running = slot["monitor"] is not None and slot["monitor"].status == "running"

    if not common.dataset_ready():
        st.warning("No dataset found. Generate one in step 1 first.")
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
        mon = TrainingMonitor()
        slot["monitor"] = mon
        slot["thread"] = run_in_background(
            train, mon, model_dir=MODELS_DIR / name, data_dir=Path(st.session_state["data_dir"]),
            model_cfg=st.session_state["model_cfg"], train_cfg=cfg,
            init_from=Path(init) if init else None)
        st.session_state["model_dir"] = str(MODELS_DIR / name)
        st.rerun()
    if c[1].button("■ Stop", disabled=not running):
        slot["monitor"].stop_event.set()
        st.toast("Stopping after the current batch …")
    c[2].caption("Training runs in the background: you can switch steps meanwhile. "
                 "Detailed logs: `tensorboard --logdir models`.")
    monitor_panel()
