"""Step 6: run the analysis and inspect individual wires."""
import io

import numpy as np
import streamlit as st
from PIL import Image

from aunw.inference import analyze_image, resize_image
from aunw.plots import maps_panel, overlay
from ui import common


def _settings():
    cfg = st.session_state["analysis_cfg"]
    with st.expander("Analysis settings", expanded=False):
        c = st.columns(3)
        cfg.auto_scale = c[0].toggle("Automatic scale", cfg.auto_scale,
                                     help="Resize so that wires are ~9 px wide, as in training.")
        cfg.target_width_px = c[0].slider("Target wire width [px]", 4.0, 16.0, cfg.target_width_px, 0.5)
        cfg.tta = c[1].toggle("Test-time augmentation (4 flips)", cfg.tta,
                              help="Averages 4 flipped predictions: more accurate, 4× slower.")
        cfg.tile_overlap = c[1].slider("Tile overlap [px]", 16, 128, cfg.tile_overlap, 16)
        cfg.max_pair_angle_deg = c[2].slider("Max. direction change in crossings [deg]", 5.0, 60.0,
                                             cfg.max_pair_angle_deg, 1.0)
        cfg.min_aspect_ratio = c[2].slider("Min. aspect ratio (else particle)", 1.0, 10.0,
                                           cfg.min_aspect_ratio, 0.5)
        c = st.columns(3)
        cfg.mask_threshold = c[0].slider("Wire probability threshold", 0.1, 0.9, cfg.mask_threshold, 0.05)
        cfg.centerline_threshold = c[1].slider("Centre-line threshold", 0.1, 0.9,
                                               cfg.centerline_threshold, 0.05)
        cfg.min_length_px = c[2].slider("Min. length [px]", 2.0, 100.0, cfg.min_length_px, 1.0)
    return cfg


def render() -> None:
    common.header(6, "Analyse", "Network prediction → wire separation → measurement of every wire.")
    model = common.active_model()
    images = st.session_state["images"]
    if model is None:
        st.warning("No network selected (blue column) – train one in step 3.")
        return
    if not images:
        st.warning("Load SEM images in step 5 first.")
        return
    cfg = _settings()
    todo = [n for n in images if n not in st.session_state["results"]]
    c = st.columns([1, 1, 3])
    run_all = c[0].button(f"▶ Analyse all ({len(images)})", type="primary")
    run_new = c[1].button(f"Analyse new ({len(todo)})", disabled=not todo)
    if run_all or run_new:
        names = list(images) if run_all else todo
        bar = st.progress(0.0)
        for i, n in enumerate(names):
            bar.progress(i / len(names), f"Analysing {n} …")
            st.session_state["results"][n] = analyze_image(model, images[n], cfg)
        bar.empty()
        st.success(f"Analysed {len(names)} image(s).")

    results = st.session_state["results"]
    if not results:
        return
    name = st.selectbox("Image", list(results))
    r = results[name]
    df = r.wires
    m = st.columns(5)
    m[0].metric("Wires found", len(df))
    m[1].metric("Complete wires", int(df["complete"].sum()) if len(df) else 0,
                help="Both tips visible → length fully observed")
    m[2].metric("Crossings resolved", f"{r.info.get('crossing_resolution', float('nan')) * 100:.0f} %")
    m[3].metric("Analysis scale", f"{r.scale:.2f}×")
    unit = "nm" if r.nm_per_px else "px"
    if len(df):
        m[4].metric(f"Median width [{unit}]", f"{df[f'width_{unit}'].median():.3g}")

    tabs = st.tabs(["Overlay", "Network output", "Wire table"])
    with tabs[0]:
        ids = st.toggle("Show wire IDs", True)
        rgb = overlay(images[name].pixels, df, r.info["paths"], show_ids=ids) if len(df) else None
        if rgb is not None:
            st.image(rgb, width="stretch",
                     caption="blue = complete · orange = end in a crossing · grey = cut by border")
            buf = io.BytesIO()
            Image.fromarray(rgb).save(buf, "PNG")
            st.download_button("⬇ Overlay (PNG)", buf.getvalue(), f"{name}_overlay.png", "image/png")
    with tabs[1]:
        st.pyplot(maps_panel(resize_image(images[name].pixels, r.scale), r.maps), width="stretch")
        mask = (r.maps["mask"] > cfg.mask_threshold).astype(np.uint8) * 255
        buf = io.BytesIO()
        Image.fromarray(mask).resize(images[name].shape[::-1], Image.NEAREST).save(buf, "PNG")
        st.download_button("⬇ Predicted mask for manual correction (step 8)", buf.getvalue(),
                           f"{name.rsplit('.', 1)[0]}_mask.png", "image/png",
                           help="Correct it in ImageJ/GIMP and use it as training data in step 8.")
    with tabs[2]:
        show = df.drop(columns=[c for c in ("image",) if c in df])
        st.dataframe(show, hide_index=True, height=420)
