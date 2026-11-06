"""AuNW Analysis - graphical user interface.

Start with:   streamlit run app.py

Layout: the blue column on the left lists the workflow steps; the main area on
the right shows the operations of the selected step. Each step lives in its
own module under ui/pages/ and exposes a single `render()` function.
"""
from __future__ import annotations

import streamlit as st

from pathlib import Path                                     # noqa: E402

_ICON = Path(__file__).parent / "assets" / "aunw.png"
st.set_page_config(page_title="AuNW Analysis", page_icon=str(_ICON) if _ICON.exists() else "🔬",
                   layout="wide", initial_sidebar_state="expanded")

from ui import common                                       # noqa: E402
from ui.pages import (analyze, finetune, images, network, overview, results,  # noqa: E402
                      synthetic_data, training, validation)

# (key, label, module, status key used for the ✓ marker)
STEPS = [
    ("overview", "Overview", overview, None),
    ("data", "1 · Synthetic data", synthetic_data, "data"),
    ("network", "2 · Network architecture", network, "network"),
    ("training", "3 · Training", training, "training"),
    ("validation", "4 · Validation", validation, "validation"),
    ("images", "5 · Load SEM images", images, "images"),
    ("analysis", "6 · Analyse", analyze, "analysis"),
    ("results", "7 · Results & export", results, "results"),
    ("finetune", "8 · Improve with real data", finetune, None),
]


def sidebar() -> str:
    status = common.workflow_status()
    with st.sidebar:
        st.markdown("## 🔬 AuNW Analysis")
        st.caption("Length & width of gold nanowires in SEM images")
        st.markdown("---")
        # The labels must never change: Streamlit re-creates a widget whose labels change,
        # and a click on the old one is lost. The ✓ of finished steps is therefore drawn
        # with CSS (::after), which does not touch the widget.
        labels = {k: lbl for k, lbl, _, _ in STEPS}
        page = st.radio("Workflow", [k for k, *_ in STEPS], format_func=labels.get,
                        key="page", label_visibility="collapsed")
        done = [i + 1 for i, (_, _, _, s) in enumerate(STEPS) if s and status.get(s)]
        if done:
            sel = ", ".join(f'section[data-testid="stSidebar"] div[role="radiogroup"] > div:nth-child({i}) '
                            f'label[data-testid="stRadioOption"] p::after' for i in done)
            st.markdown(f"<style>{sel} {{content: '  ✓';}}</style>", unsafe_allow_html=True)
        st.markdown("---")
        models = common.available_models()
        if models:
            names = [m.name for m in models]
            current = st.session_state.get("model_dir")
            idx = names.index(current.split("/")[-1]) if current and current.split("/")[-1] in names else 0
            choice = st.selectbox("Active network", names, index=idx,
                                  help="Used for analysis, validation and visualisation.")
            st.session_state["model_dir"] = str(models[names.index(choice)])
        else:
            st.info("No trained network yet – see step 3.")
        slot = common.trainer_slot()
        if slot["process"] is not None and slot["process"].poll() is None:
            st.markdown("⏳ **Training running** – progress in step 3")
        st.markdown("---")
        st.caption("All data stay on this computer. No telemetry.")
    return page


def main() -> None:
    common.init_state()
    common.apply_style()
    page = sidebar()
    module = {k: m for k, _, m, _ in STEPS}[page]
    module.render()


main()
