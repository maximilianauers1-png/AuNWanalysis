"""Step 2: choose the architecture and visualise the neural network."""
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from aunw.dataset import list_test_images
from aunw.model import build_unet, layer_table, receptive_field
from aunw.plots import feature_maps, unet_diagram
from ui import common

PRESETS = {
    "CPU – fast (16 filters, ≈2 M parameters)": 16,
    "Standard – recommended on a GPU (24 filters, ≈4.7 M)": 24,
    "Large (32 filters, ≈8 M)": 32,
}


@st.cache_resource(show_spinner="Building network …")
def _built(base: int, depth: int, dropout: float):
    from aunw.config import ModelConfig
    return build_unet(ModelConfig(base_filters=base, depth=depth, dropout=dropout))


def render() -> None:
    common.header(2, "Network architecture",
                  "Residual U-Net with three output heads. Configure it here and look inside.")
    cfg = st.session_state["model_cfg"]
    c1, c2, c3 = st.columns([2, 1, 1])
    preset = c1.selectbox("Size", list(PRESETS), index=list(PRESETS.values()).index(cfg.base_filters)
                          if cfg.base_filters in PRESETS.values() else 1)
    cfg.base_filters = PRESETS[preset]
    cfg.depth = c2.slider("Depth (down-sampling steps)", 2, 5, cfg.depth)
    cfg.dropout = c3.slider("Spatial dropout (deep levels)", 0.0, 0.5, cfg.dropout, 0.05)

    model = _built(cfg.base_filters, cfg.depth, cfg.dropout)
    m = st.columns(4)
    m[0].metric("Trainable parameters", f"{model.count_params() / 1e6:.2f} M")
    m[1].metric("Receptive field", f"{receptive_field(cfg)} px")
    m[2].metric("Bottleneck resolution", f"{cfg.tile_size // 2**cfg.depth} px")
    m[3].metric("Filters per level", "-".join(map(str, cfg.filters)))

    tabs = st.tabs(["Diagram", "Layer table", "Keras graph", "Look inside a trained network"])
    with tabs[0]:
        common.show_figure(unet_diagram(cfg), "unet_architecture", width="stretch")
        st.markdown(
            "**Reading the diagram.** The *encoder* (blue) halves the resolution at every level "
            "and doubles the number of feature channels: it learns *what* is in the image. The "
            "*decoder* (green) restores full resolution; the dashed skip connections hand over "
            "fine detail lost by pooling, which is essential for exact wire edges (width!). "
            f"With a receptive field of {receptive_field(cfg)} px each output pixel sees enough "
            "context to follow wires through crossings.")
    with tabs[1]:
        df = pd.DataFrame(layer_table(model))
        st.dataframe(df, width="stretch", height=420, hide_index=True)
        st.caption(f"{len(df)} layers · {df.parameters.sum():,} parameters")
    with tabs[2]:
        try:
            import keras
            png = "/tmp/aunw_model_graph.png"
            keras.utils.plot_model(model, png, show_shapes=True, dpi=60, rankdir="TB")
            st.image(png)
        except Exception:
            st.info("The full Keras graph needs `pydot` and Graphviz (`pip install pydot` and "
                    "install Graphviz). The diagram tab shows the same structure.")
    with tabs[3]:
        trained = common.active_model()
        tests = list_test_images(st.session_state["data_dir"])
        if trained is None or not tests:
            st.info("Needs a trained network (step 3) and synthetic test images (step 1).")
            return
        img = cv2.imread(str(tests[0]), cv2.IMREAD_GRAYSCALE)[:256, :256].astype(np.float32) / 255
        names = [l.name for l in trained.layers if l.name.endswith(("_out", "_relu1")) or l.name in
                 ("mask", "centerline", "orientation")]
        layer = st.selectbox("Layer", names, index=0)
        c = st.columns([1, 3])
        c[0].image(img, caption="Input tile", clamp=True)
        with c[1]:
            st.pyplot(feature_maps(trained, img, layer), width="stretch")
        st.caption("Feature maps with the highest variance. Early layers respond to edges and "
                   "noise; deep layers to whole wires, crossings and particles.")
