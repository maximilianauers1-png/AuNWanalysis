"""Step 8: improve the network with real, hand-annotated SEM images (fine-tuning)."""
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from aunw.config import MODELS_DIR
from aunw.dataset import tiles_from_annotations
from aunw.inference import estimate_scale, resize_image
from aunw.io import load_image
from aunw.training import model_file
from ui import common
from ui.pages.training import is_running, launch, monitor_panel


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

    if st.button("▶ Fine-tune", type="primary", disabled=is_running()):
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
            try:
                real_train = tiles_from_annotations(imgs[:k], masks[:k])
                real_val = tiles_from_annotations(imgs[k:], masks[k:])
            except ValueError as exc:
                st.error(str(exc))
                return
        out = MODELS_DIR / name
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "real_tiles.npz", **{f"train_{k}": v for k, v in real_train.items()},
                            **{f"val_{k}": v for k, v in real_val.items()})
        args = ["finetune", "--base", str(model_file(Path(st.session_state["model_dir"]))),
                "--out", str(out), "--real", str(out / "real_tiles.npz"),
                "--synthetic", str(Path(st.session_state["data_dir"]) / "train.npz"),
                "--epochs", str(int(epochs)), "--lr", str(lr), "--real-fraction", str(real_frac)]
        args += [] if freeze else ["--no-freeze"]
        launch(args, out)
        st.session_state["model_dir"] = str(out)
        st.toast(f"Fine-tuning started: {len(real_train['image'])} real training tiles, "
                 f"{len(real_val['image'])} validation tiles")
    st.caption("Note: real annotations carry no orientation label, so the validation angle error "
               "is not defined for them and is shown as 0.")
    monitor_panel()
