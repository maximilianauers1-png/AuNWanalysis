"""Step 8: improve the network with real, hand-annotated SEM images (fine-tuning)."""
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from aunw.config import MODELS_DIR, TrainConfig
from aunw.dataset import load_split, tiles_from_annotations
from aunw.inference import estimate_scale, resize_image
from aunw.io import load_image
from aunw.training import TrainingMonitor, finetune, model_file, run_in_background
from ui import common
from ui.pages.training import _monitor


def render() -> None:
    common.header(8, "Improve with real data",
                  "Fine-tune the synthetic-trained network on a few annotated real micrographs.")
    st.markdown(
        "**Workflow.** (1) Analyse a real image in step 6 and download *Predicted mask for manual "
        "correction*. (2) Correct the mask in ImageJ/Fiji or GIMP (white = wire, black = "
        "background). (3) Upload images + corrected masks here (`name.tif` + `name_mask.png`). "
        "(4) Fine-tune. (5) Compare old and new network in step 4.")
    with st.expander("What happens during fine-tuning?"):
        st.markdown(
            "- The **encoder is frozen**: the general feature extractors learned from thousands "
            "of synthetic images are kept; only the decoder adapts to your microscope.\n"
            "- Each batch mixes **real and synthetic tiles** (default 50/50), which prevents "
            "*catastrophic forgetting*.\n"
            "- Small learning rate, early stopping on **held-out real images** (split by image, "
            "never by tile – neighbouring tiles of one image are not independent).")

    base = common.active_model()
    if base is None:
        st.warning("Select a trained network first.")
        return
    files = st.file_uploader("Images and corrected masks", type=["tif", "tiff", "png", "jpg", "jpeg"],
                             accept_multiple_files=True, key="ft_files")
    by_stem = {f.name.rsplit(".", 1)[0]: f for f in files or []}
    pairs = [(s, f, by_stem[f"{s}_mask"]) for s, f in by_stem.items() if f"{s}_mask" in by_stem]
    st.caption(f"{len(pairs)} image/mask pair(s) found.")
    if len(pairs) < 2:
        st.info("At least 2 annotated images are needed (one is held out for validation).")
        return

    c = st.columns(4)
    n_val = c[0].number_input("Images held out for validation", 1, len(pairs) - 1, max(1, len(pairs) // 4))
    epochs = c[1].number_input("Max. epochs", 1, 200, 20)
    lr = c[2].select_slider("Learning rate", [1e-5, 3e-5, 1e-4, 3e-4], 1e-4, format_func=lambda v: f"{v:.0e}")
    real_frac = c[3].slider("Fraction of real tiles per batch", 0.1, 0.9, 0.5, 0.1)
    freeze = st.toggle("Freeze encoder", True)
    name = st.text_input("Name of the fine-tuned model",
                         f"{Path(st.session_state['model_dir']).name}_ft_{datetime.now():%Y%m%d_%H%M}")

    slot = common.trainer_slot()
    running = slot["monitor"] is not None and slot["monitor"].status == "running"
    if st.button("▶ Fine-tune", type="primary", disabled=running):
        cfg = st.session_state["analysis_cfg"]
        imgs, masks = [], []
        with st.spinner("Preparing tiles at network scale …"):
            for _, f, m in pairs:
                img = load_image(f.getvalue(), f.name).pixels
                mask = np.asarray(Image.open(m).convert("L")).astype(np.float32) / 255
                s = estimate_scale(base, img, cfg) if cfg.auto_scale else 1.0
                imgs.append(resize_image(img, s))
                masks.append(resize_image(mask, s) > 0.5)
            k = len(pairs) - int(n_val)
            real_train = tiles_from_annotations(imgs[:k], masks[:k])
            real_val = tiles_from_annotations(imgs[k:], masks[k:])
            synth = load_split(Path(st.session_state["data_dir"]) / "train.npz")
        tcfg = TrainConfig(epochs=int(epochs), learning_rate=lr, batch_size=8, early_stop_patience=6)
        mon = TrainingMonitor()
        slot["monitor"] = mon
        slot["thread"] = run_in_background(
            finetune, mon, base_model=model_file(Path(st.session_state["model_dir"])),
            model_dir=MODELS_DIR / name, real_train=real_train, real_val=real_val, synthetic=synth,
            train_cfg=tcfg, freeze_encoder=freeze, real_fraction=real_frac)
        st.session_state["model_dir"] = str(MODELS_DIR / name)
        st.toast(f"Fine-tuning started: {len(real_train['image'])} real training tiles, "
                 f"{len(real_val['image'])} validation tiles")
    _monitor()
