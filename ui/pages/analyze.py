"""Step 6: run the analysis and inspect individual wires."""
import io
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from aunw.inference import analyze_image, resize_image
from aunw import report
from aunw.plots import figure_bytes, maps_panel
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


def _measured_wires(name: str, r, images: dict) -> None:
    """Map with every wire numbered and labelled, the wire table, and all exports."""
    if r.wires.empty:
        return
    image = images[name].pixels
    stem = name.rsplit(".", 1)[0]
    table = report.wire_table(r)
    c = st.columns([2, 3])
    values = c[0].toggle("Show length and width on the map", True,
                         help="Off: wire numbers only - more readable for very dense images.")
    c[1].caption("Select rows in the table below to highlight those wires on the map. "
                 "Coordinates: origin top-left, x right, y down (as in ImageJ). "
                 "L ≥ … = visible length of a wire that is not complete (lower bound).")

    show_cols = [col for col in ("wire_id", "status", "length_nm", "width_nm", "length_px",
                                 "width_px", "start_x_um", "start_y_um", "end_x_um", "end_y_um",
                                 "start_x_px", "start_y_px", "end_x_px", "end_y_px",
                                 "aspect_ratio", "angle_deg") if col in table]
    sel = st.session_state.get(f"wiresel_{name}")
    rows = sel.selection.rows if sel is not None and hasattr(sel, "selection") else []
    highlight = set(table.iloc[rows]["wire_id"]) if rows else None

    fig = report.wire_map(image, r, values, highlight)
    png = figure_bytes(fig, "png")
    st.image(png, width="stretch")
    st.dataframe(table[show_cols].round(3), hide_index=True, height=320, on_select="rerun",
                 selection_mode="multi-row", key=f"wiresel_{name}")

    _save_section(name, image, r, images)

    with st.expander("Download individual files (browser downloads go to your Downloads folder)"):
        e = st.columns(6)
        e[0].download_button("⬇ Map (PNG)", png, f"{stem}_map_{'values' if values else 'ids'}.png",
                             "image/png", key=f"png_{name}")
        e[1].download_button("⬇ Map (PDF)", figure_bytes(fig, "pdf"), f"{stem}_map.pdf",
                             "application/pdf", key=f"pdf_{name}")
        e[2].download_button("⬇ Wires (CSV)", table.to_csv(index=False).encode("utf-8-sig"),
                             f"{stem}_wires.csv", "text/csv", key=f"csv_{name}")
        e[3].download_button("⬇ Path points (CSV)",
                             report.path_points(r).to_csv(index=False).encode("utf-8-sig"),
                             f"{stem}_path_points.csv", "text/csv", key=f"pts_{name}")
        cache = st.session_state.setdefault("export_cache", {})
        if e[4].button("Prepare Excel", key=f"prepx_{name}"):
            with st.spinner("Building workbook with maps …"):
                cache[("xlsx", name)] = report.image_excel(image, r)
        if ("xlsx", name) in cache:
            e[4].download_button("⬇ Excel", cache[("xlsx", name)], f"{stem}_wires.xlsx",
                                 "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                 key=f"xlsx_{name}")
        if e[5].button("Prepare ZIP", key=f"prepz_{name}"):
            with st.spinner("Building ZIP …"):
                cache[("zip", name)] = report.image_zip(image, r)
        if ("zip", name) in cache:
            e[5].download_button("⬇ ZIP", cache[("zip", name)], f"{stem}_wires.zip",
                                 "application/zip", key=f"zip_{name}")


def _save_section(name: str, image, r, images: dict) -> None:
    """Save the result files straight into a folder on this computer."""
    st.markdown("**Save results** – tables (CSV + Excel with the maps inside) and maps (PNG + PDF)")
    folder = st.text_input("Save folder", str(st.session_state.get("save_folder",
                                                                   report.default_save_folder())),
                           help="Each save creates a new sub-folder named with date and time.")
    st.session_state["save_folder"] = folder
    results = st.session_state["results"]
    c = st.columns([1, 1, 1, 2])
    target = None
    if c[0].button("💾 Save this image", type="primary", key=f"save_{name}"):
        with st.spinner("Writing files …"):
            files = {f"{name.rsplit('.', 1)[0]}/{k}": v
                     for k, v in report.files_for_image(image, r).items()}
            target = report.save_files(files, Path(folder))
    if c[1].button(f"💾 Save all {len(results)} images", key="save_all"):
        with st.spinner(f"Writing files for {len(results)} images …"):
            target = report.save_files(report.all_files(
                list(results.values()), {n: im.pixels for n, im in images.items()}), Path(folder))
    if target is not None:
        st.session_state["last_saved"] = str(target)
    last = st.session_state.get("last_saved")
    if last:
        st.success(f"Saved to **{last}**")
        if c[2].button("📂 Open folder", key="open_folder"):
            report.open_folder(Path(last))


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
        st.session_state["export_cache"] = {}                 # results changed: rebuild exports
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
    m[3].metric("Areal coverage", f"{r.info.get('coverage', float('nan')) * 100:.0f} %",
                help="Fraction of the image covered by wires.")
    unit = "nm" if r.nm_per_px else "px"
    if len(df):
        m[4].metric(f"Median width [{unit}]", f"{df[f'width_{unit}'].median():.3g}")
    f = r.nm_per_px or 1.0
    l_complete = df.loc[df["complete"], "length_px"].mean() * f if len(df) else float("nan")
    l_stereo = r.info.get("stereological_mean_length_px", float("nan")) * f
    msg = (f"Mean length: {l_complete:.4g} {unit} (complete wires) vs. {l_stereo:.4g} {unit} "
           f"(stereological, 2·total length / number of ends) · analysis scale {r.scale:.2f}×")
    rel = r.info.get("reliability", "")
    if len(df) == 0:
        st.warning("No wires detected. Check that the right network is active (blue column), "
                   "that the image shows nanowires, and try the automatic scale or a different "
                   "target width in the analysis settings.")
    elif rel == "good":
        st.success(f"✓ Coverage is moderate – length and width statistics are reliable. {msg}")
    elif rel == "caution":
        st.warning(f"⚠️ Dense mesh – some wire ends are hidden; length statistics may be biased "
                   f"towards short wires. Width is reliable. {msg}")
    else:
        st.error("✗ Very dense mesh – most wire ends are hidden under other wires, so wire "
                 "lengths cannot be measured in this image. The **width is reliable**. "
                 f"(analysis scale {r.scale:.2f}×)")

    w_an = r.info.get("width_px_global", float("nan"))
    if w_an < 5:
        st.warning(f"⚠️ The wires are only about {w_an:.1f} px wide at the analysis scale. Widths below "
                   "~5 px are at the resolution limit (± 0.5 px is already ± 10 %). Record images at "
                   "higher magnification for accurate widths.")
    if len(df) and df["touches_border"].mean() > 0.6:
        st.info(f"ℹ️ {df['touches_border'].mean():.0%} of the wires leave the field of view, so their "
                "length cannot be measured in this image (the width can). For length statistics "
                "use images at lower magnification in which whole wires are visible – and keep "
                "the high-magnification images for the width.")
    tabs = st.tabs(["Measured wires (map + table)", "Network output"])
    with tabs[0]:
        _measured_wires(name, r, images)
    with tabs[1]:
        st.pyplot(maps_panel(resize_image(images[name].pixels, r.scale), r.maps), width="stretch")
        mask = (r.maps["mask"] > cfg.mask_threshold).astype(np.uint8) * 255
        buf = io.BytesIO()
        Image.fromarray(mask).resize(images[name].shape[::-1], Image.NEAREST).save(buf, "PNG")
        st.download_button("⬇ Predicted mask for manual correction (step 8)", buf.getvalue(),
                           f"{name.rsplit('.', 1)[0]}_mask.png", "image/png",
                           help="Correct it in ImageJ/GIMP and use it as training data in step 8.")
