"""Overview: what the software does and where you are in the workflow."""
import streamlit as st

from ui import common


def render() -> None:
    common.header("Welcome", "Gold-nanowire mesh analysis",
                  "Individual length and width of every nanowire in SEM images, "
                  "measured by a neural network trained on synthetic micrographs.")
    status = common.workflow_status()

    st.markdown("#### How it works")
    c = st.columns(4)
    with c[0]:
        common.card("① Learn from synthetic SEM images",
                    "A physics-based renderer creates unlimited chaotic AuNW meshes with exact labels.")
    with c[1]:
        common.card("② Segment with a U-Net",
                    "The network predicts per pixel: wire, wire centre-line and wire orientation.")
    with c[2]:
        common.card("③ Separate the wires",
                    "A skeleton graph follows each wire straight through crossings.")
    with c[3]:
        common.card("④ Measure & report",
                    "Sub-pixel widths, lengths, statistics, fits and exportable figures/tables.")

    st.markdown("#### Your progress")
    rows = [("Synthetic training data", "data", "Step 1"),
            ("Trained network available", "network", "Step 3"),
            ("Network validated", "validation", "Step 4"),
            ("SEM images loaded", "images", "Step 5"),
            ("Images analysed", "analysis", "Step 6")]
    for text, key, step in rows:
        ok = status[key]
        badge = '<span class="badge-ok">✓ done</span>' if ok else '<span class="badge-todo">○ open</span>'
        st.markdown(f"{badge} &nbsp; {text} <span style='color:#888'>({step})</span>",
                    unsafe_allow_html=True)

    st.markdown("#### Quick start for analysing images")
    st.markdown(
        "1. Choose a trained network in the blue column (*Active network*).\n"
        "2. **Step 5** – upload TIFF/JPEG micrographs and check the nm/pixel calibration.\n"
        "3. **Step 6** – run the analysis and inspect the overlay.\n"
        "4. **Step 7** – download tables (CSV/Excel) and figures (PNG/SVG/PDF).\n\n"
        "Steps 1–4 are only needed to (re)train the network; step 8 adapts it to your own "
        "microscope with a few hand-annotated images.")
    with st.expander("Colour code of the overlays"):
        st.markdown("- **Blue** – complete wire (both tips visible): used for length statistics\n"
                    "- **Orange** – wire whose end disappears in a crossing: used for width only\n"
                    "- **Grey** – wire cut by the image border: used for width only")
