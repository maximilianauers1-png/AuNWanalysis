"""Step 1: configure, preview and generate synthetic training data."""
from pathlib import Path

import numpy as np
import streamlit as st

from aunw.dataset import generate_dataset, load_split
from aunw.synthetic import IGNORE_ANGLE, render_sample
from ui import common


def _config_form():
    cfg = st.session_state["synthetic_cfg"]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Mesh geometry**")
        cfg.n_wires = common.range_slider("Wires per 256² tile", 1, 100, cfg.n_wires, 1,
                                          "Mesh density. Chosen uniformly per image.")
        cfg.length_median = common.range_slider("Median wire length [px]", 10.0, 600.0,
                                                cfg.length_median, 5.0)
        cfg.length_sigma = common.range_slider("Length spread σ (log-normal)", 0.05, 1.2,
                                               cfg.length_sigma, 0.05)
        cfg.max_bend_deg = st.slider("Max. bending angle [deg]", 0.0, 90.0, cfg.max_bend_deg, 1.0)
    with c2:
        st.markdown("**Wire width**")
        cfg.width_mean = common.range_slider("Mean width [px]", 2.0, 30.0, cfg.width_mean, 0.5)
        cfg.width_cv = common.range_slider("Width variation (CV)", 0.0, 0.5, cfg.width_cv, 0.01)
        st.markdown("**Distractors**")
        cfg.particle_prob = st.slider("Probability of Au particles", 0.0, 1.0, cfg.particle_prob, 0.05)
        cfg.pore_prob = st.slider("Probability of porous membrane", 0.0, 1.0, cfg.pore_prob, 0.05)
    with c3:
        st.markdown("**SEM imaging**")
        cfg.edge_gain = common.range_slider("Edge-effect strength", 0.0, 1.5, cfg.edge_gain, 0.05,
                                            "Bright wire edges from the SE yield ∝ 1/cos φ")
        cfg.blur_sigma = common.range_slider("Probe blur σ [px]", 0.0, 3.0, cfg.blur_sigma, 0.1)
        cfg.photons = common.range_slider("Electron dose (higher = less noise)", 5.0, 500.0,
                                          cfg.photons, 5.0)
        cfg.jpeg_prob = st.slider("Probability of JPEG artefacts", 0.0, 1.0, cfg.jpeg_prob, 0.05)
    return cfg


def _preview(cfg) -> None:
    seed = st.session_state.setdefault("preview_seed", 0)
    if st.button("🎲 New random examples"):
        st.session_state["preview_seed"] = seed = seed + 1
    rng = np.random.default_rng(seed)
    samples = [render_sample(rng, cfg) for _ in range(3)]
    for s in samples:
        ang = s["angle"].astype(float)
        ang_rgb = np.zeros(ang.shape + (3,))
        valid = ang != IGNORE_ANGLE
        import matplotlib.cm as cm
        ang_rgb[valid] = cm.hsv(ang[valid] / 180)[:, :3]
        cols = st.columns(4)
        cols[0].image(s["image"], caption=f"SEM image ({len(s['wires'])} wires)", clamp=True)
        cols[1].image(s["mask"] * 255, caption="Label: wire mask")
        cols[2].image(s["centerline"] * 255, caption="Label: centre-lines")
        cols[3].image(ang_rgb, caption="Label: orientation (hue), black = ignored", clamp=True)


def render() -> None:
    common.header(1, "Synthetic training data",
                  "Physically motivated SEM renderings of chaotic nanowire meshes with exact labels.")
    with st.expander("Why synthetic data?", expanded=False):
        st.markdown(
            "Labelling every pixel of hundreds of overlapping wires by hand is not feasible. "
            "The renderer draws each wire as a capsule along a circular arc and computes "
            "**exact** masks, centre-lines and orientations. The SEM look is simulated: bright "
            "edges (secondary-electron yield ∝ 1/cos φ), detector shadow, probe blur, Poisson "
            "shot noise, scan-line noise and JPEG artefacts. All parameters are drawn randomly "
            "from the ranges below (*domain randomisation*), so the network learns the wire "
            "shape, not one particular imaging condition.")

    cfg = _config_form()
    st.markdown("### Preview")
    _preview(cfg)

    st.markdown("### Generate dataset")
    c = st.columns(4)
    n_train = c[0].number_input("Training tiles", 100, 50000, 1500, 100)
    n_val = c[1].number_input("Validation tiles", 50, 10000, 250, 50,
                              help="Generated with a different random seed → independent.")
    n_test = c[2].number_input("Full-size test images", 0, 200, 12, 1,
                               help="768×768 images with per-wire ground truth (step 4).")
    out = c[3].text_input("Output folder", st.session_state["data_dir"])
    if st.button("⚙️ Generate dataset", type="primary"):
        bar = st.progress(0.0, "Starting …")
        generate_dataset(Path(out), cfg, int(n_train), int(n_val), int(n_test),
                         progress=lambda name, f: bar.progress(f, f"Rendering {name} set … {f:.0%}"))
        st.session_state["data_dir"] = out
        bar.empty()
        st.success(f"Dataset written to {out}")

    if common.dataset_ready(out):
        d = Path(out)
        sizes = {k: len(load_split(d / f"{k}.npz")["image"]) for k in ("train", "val")}
        n_t = len(list((d / "test").glob("test_*[0-9].png")))
        st.info(f"Dataset in **{out}**: {sizes['train']} training tiles, {sizes['val']} "
                f"validation tiles, {n_t} test images.")
