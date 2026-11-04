"""Step 4: test the network – on synthetic ground truth and on real SEM data."""
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from aunw.evaluation import evaluate_real, evaluate_synthetic
from aunw.export import wires_table
from aunw.io import load_image
from aunw.plots import histogram_fit, parity_plot
from aunw.stats import compare_samples
from ui import common


def _synthetic(model) -> None:
    st.markdown("Runs the **complete pipeline** (tiling → network → wire separation → "
                "measurement) on full-size synthetic test images whose true wire lengths and "
                "widths are known.")
    n = st.slider("Number of test images", 1, 50, 12)
    if st.button("▶ Evaluate on synthetic test images", type="primary"):
        bar = st.progress(0.0)
        st.session_state["evaluation"] = evaluate_synthetic(
            model, st.session_state["data_dir"], st.session_state["analysis_cfg"], n,
            progress=lambda f: bar.progress(f, f"{f:.0%}"))
        bar.empty()
    ev = st.session_state.get("evaluation")
    if ev is None:
        return
    s = ev["summary"]
    m = st.columns(4)
    m[0].metric("Mask Dice", f"{s['mask_dice']:.3f}")
    m[1].metric("Centre-line Dice", f"{s['centerline_dice']:.3f}")
    m[2].metric("Orientation error", f"{s['angle_error_deg']:.1f}°")
    m[3].metric("Crossings resolved", f"{s['crossing_resolution'] * 100:.0f} %")
    m = st.columns(4)
    m[0].metric("Length: median error", f"{s['length_median_rel_error_%']:+.1f} %",
                help="Complete wires matched to ground truth")
    m[1].metric("Length: within ±10 %", f"{s['length_within_10%_fraction'] * 100:.0f} %")
    m[2].metric("Width: median error", f"{s['width_median_rel_error_%']:+.1f} %")
    m[3].metric("Width: within ±10 %", f"{s['width_within_10%_fraction'] * 100:.0f} %")
    st.caption(f"Population level (all wires pooled): length bias mean "
               f"{s.get('population_mean_length_bias_%', float('nan')):+.1f} % / median "
               f"{s.get('population_median_length_bias_%', float('nan')):+.1f} %, width bias mean "
               f"{s.get('population_mean_width_bias_%', float('nan')):+.1f} % / median "
               f"{s.get('population_median_width_bias_%', float('nan')):+.1f} %. In very dense meshes complete "
               "wires under-represent long wires (long wires cross more often and are more "
               "likely to lose an end in a crossing) – keep this in mind when comparing samples "
               "of very different density.")
    c = st.columns(2)
    comp, wires = ev["complete"], ev["wires"]
    with c[0]:
        common.show_figure(parity_plot(comp.true_length, comp.length, "Length"), "parity_length")
    with c[1]:
        common.show_figure(parity_plot(wires.true_width, wires.width, "Width"), "parity_width")
    with st.expander("Per-image details"):
        st.dataframe(ev["segmentation"].merge(ev["per_image"], on="image"), hide_index=True)


def _real_masks(model) -> None:
    st.markdown("Upload real SEM images **and** binary masks (white = wire) with the same file "
                "name plus `_mask` (e.g. `sample1.tif` + `sample1_mask.png`). Masks can be drawn "
                "in ImageJ/Fiji, GIMP or exported from step 6 and corrected.")
    files = st.file_uploader("Images and masks", type=["tif", "tiff", "png", "jpg", "jpeg"],
                             accept_multiple_files=True, key="val_real")
    if not files:
        return
    by_stem = {f.name.rsplit(".", 1)[0]: f for f in files}
    pairs = [(stem, f, by_stem[f"{stem}_mask"]) for stem, f in by_stem.items() if f"{stem}_mask" in by_stem]
    if not pairs:
        st.warning("No image/mask pairs found (expected `name` + `name_mask`).")
        return
    if st.button(f"▶ Evaluate {len(pairs)} annotated image(s)"):
        imgs = [load_image(f.getvalue(), f.name).pixels for _, f, _ in pairs]
        masks = [np.asarray(Image.open(m).convert("L")) > 127 for _, _, m in pairs]
        res = evaluate_real(model, imgs, masks, [s for s, _, _ in pairs], st.session_state["analysis_cfg"])
        st.dataframe(res, hide_index=True)
        st.metric("Mean Dice on real images", f"{res.mask_dice.mean():.3f}")


def _manual(model) -> None:
    st.markdown("Compare the automatic results of step 6 with **manual measurements** (e.g. "
                "ImageJ line tool). Upload a CSV with a column `length` and/or `width` "
                "(same unit as selected below).")
    wires = wires_table(st.session_state["results"].values())
    if wires.empty:
        st.info("Analyse images in step 6 first.")
        return
    f = st.file_uploader("Manual measurements (CSV)", type=["csv"], key="manual_csv")
    if f is None:
        return
    manual = pd.read_csv(f)
    unit = "nm" if "length_nm" in wires else "px"
    st.caption(f"Automatic values are compared in **{unit}**.")
    for q, subset in (("length", wires[wires.complete]), ("width", wires)):
        if q not in manual:
            continue
        auto = subset[f"{q}_{unit}"]
        res = compare_samples(auto, manual[q])
        st.markdown(f"**{q.capitalize()}**")
        st.dataframe(pd.DataFrame([res]).T.rename(columns={0: "value"}))
        verdict = ("no significant difference" if res["KS_p"] > 0.05 else "distributions differ")
        st.caption(f"Kolmogorov–Smirnov p = {res['KS_p']:.3g} → {verdict} (α = 0.05).")
        c = st.columns(2)
        c[0].pyplot(histogram_fit(auto, f"{q} (automatic)", unit), width="stretch")
        c[1].pyplot(histogram_fit(manual[q], f"{q} (manual)", unit), width="stretch")


def render() -> None:
    common.header(4, "Validation", "Is the network accurate? Quantify it before trusting results.")
    model = common.active_model()
    if model is None:
        st.warning("Select or train a network first (step 3).")
        return
    tabs = st.tabs(["Synthetic ground truth", "Real images with masks", "Manual measurements"])
    with tabs[0]:
        _synthetic(model)
    with tabs[1]:
        _real_masks(model)
    with tabs[2]:
        _manual(model)
