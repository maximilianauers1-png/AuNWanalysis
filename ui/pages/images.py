"""Step 5: load SEM images, crop the info bar and calibrate nm/pixel."""
from pathlib import Path

import streamlit as st

from aunw.io import crop_databar, detect_databar, list_images, load_image
from ui import common


def _add(sem) -> None:
    st.session_state["raw_images"][sem.name] = sem
    rows = detect_databar(sem.pixels)
    st.session_state.setdefault("crop", {})[sem.name] = rows
    st.session_state["images"][sem.name] = crop_databar(sem, rows)


def _calibration(name: str) -> None:
    raw = st.session_state["raw_images"][name]
    img = st.session_state["images"][name]
    c = st.columns([3, 2])
    with c[1]:
        st.markdown("**Information bar**")
        h = raw.pixels.shape[0]
        rows = st.slider("Rows to crop at the bottom", 0, h // 2,
                         st.session_state["crop"].get(name, 0), key=f"crop_{name}",
                         help="Suggested automatically; the info bar must not be analysed.")
        st.session_state["crop"][name] = rows

        st.markdown("**Calibration**")
        if raw.nm_per_px:
            st.success(f"From TIFF metadata: {raw.nm_per_px:.4g} nm/px")
        modes = ["Metadata", "nm per pixel", "Scale bar", "None (pixels)"]
        mode = st.radio("Source", modes if raw.nm_per_px else modes[1:], key=f"calmode_{name}",
                        horizontal=True)
        nm = raw.nm_per_px
        if mode == "nm per pixel":
            nm = st.number_input("nm / pixel", 0.0001, 1e5, float(raw.nm_per_px or 1.0),
                                 format="%.4f", key=f"nmpx_{name}")
        elif mode == "Scale bar":
            a, b = st.columns(2)
            bar_nm = a.number_input("Scale bar length [nm]", 1.0, 1e7, 1000.0, key=f"barnm_{name}")
            bar_px = b.number_input("Scale bar length [px]", 1.0, 1e5, 200.0, key=f"barpx_{name}",
                                    help="Measure it e.g. in ImageJ, or read the bar width.")
            nm = bar_nm / bar_px
            st.caption(f"→ {nm:.4g} nm/px")
        elif mode == "None (pixels)":
            nm = None
        img = crop_databar(raw, rows)
        img.nm_per_px = nm
        img.calibration_source = mode
        st.session_state["images"][name] = img
        if st.button("Remove image", key=f"rm_{name}"):
            for k in ("images", "raw_images", "results"):
                st.session_state[k].pop(name, None)
            st.rerun()
    with c[0]:
        st.image(img.pixels, caption=f"{name} · {img.shape[1]}×{img.shape[0]} px (analysed region)",
                 clamp=True, width="stretch")


def render() -> None:
    common.header(5, "Load SEM images", "TIFF (8/16 bit, with or without metadata), JPEG or PNG.")
    tabs = st.tabs(["Upload files", "Read a folder on this computer"])
    with tabs[0]:
        files = st.file_uploader("Drop SEM images here", type=["tif", "tiff", "jpg", "jpeg", "png"],
                                 accept_multiple_files=True)
        for f in files or []:
            if f.name not in st.session_state["raw_images"]:
                _add(load_image(f.getvalue(), f.name))
    with tabs[1]:
        folder = st.text_input("Folder path")
        if st.button("Load all images of this folder") and folder:
            paths = list_images(Path(folder))
            for p in paths:
                _add(load_image(p))
            st.success(f"Loaded {len(paths)} image(s).")

    names = list(st.session_state["images"])
    if not names:
        st.info("No images loaded yet.")
        return
    st.markdown(f"### {len(names)} image(s) loaded")
    if len(names) > 1 and st.button("Apply calibration of the first image to all images"):
        ref = st.session_state["images"][names[0]]
        for n in names[1:]:
            st.session_state[f"calmode_{n}"] = "nm per pixel"
            st.session_state[f"nmpx_{n}"] = float(ref.nm_per_px or 1.0)
        st.rerun()
    name = st.selectbox("Inspect / calibrate", names)
    _calibration(name)
    st.dataframe([{"image": n, "size": f"{i.shape[1]}×{i.shape[0]}",
                   "nm/px": f"{i.nm_per_px:.4g}" if i.nm_per_px else "–",
                   "calibration": i.calibration_source}
                  for n, i in st.session_state["images"].items()], hide_index=True)
