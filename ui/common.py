"""Shared GUI helpers: session state, cached resources, styling, small widgets."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from aunw.config import (DATA_DIR, MODELS_DIR, AnalysisConfig, ModelConfig, SyntheticConfig,
                         TrainConfig)
from aunw.plots import figure_bytes

# --------------------------------------------------------------------------- styling
CSS = """
<style>
/* ---- left step column: deep blue -------------------------------------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d2c6b 0%, #123f94 55%, #1a56c4 100%);
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] [data-testid="stHeadingWithActionElements"] * { color: #eef3ff !important; }
section[data-testid="stSidebar"] [data-testid="stAlert"] * { color: #0b0b0b !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.18); }

/* step list = radio group without the radio circles */
section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 2px; width: 100%; }
section[data-testid="stSidebar"] div[role="radiogroup"] > div { width: 100%; }
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] {
    padding: 9px 12px; border-radius: 8px; margin: 0; width: 100%;
    border-left: 3px solid transparent; transition: background .15s; cursor: pointer;
}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] > div > div:first-child { display: none; }
section[data-testid="stSidebar"] label[data-testid="stRadioOption"]:hover { background: rgba(255,255,255,.10); }
section[data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"] {
    background: rgba(255,255,255,.20); border-left: 3px solid #8fc1ff;
}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"] p { font-weight: 700; }
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] p { font-size: .98rem; }

/* ---- main area -------------------------------------------------------- */
.block-container { padding-top: 3.2rem; max-width: 1400px; }
.step-header { border-bottom: 1px solid #e3e8f2; margin-bottom: 1rem; padding-bottom: .4rem; }
.step-header .kicker { color: #1a56c4; font-size: .8rem; font-weight: 700; letter-spacing: .08em;
                       text-transform: uppercase; }
.step-header h1 { font-size: 1.75rem; margin: .1rem 0 .2rem 0; }
.step-header p { color: #52514e; margin: 0; }
.card { border: 1px solid #e3e8f2; border-radius: 10px; padding: 14px 16px; background: #fafcff;
        height: 100%; }
.card h4 { margin: 0 0 .3rem 0; color: #0d2c6b; }
.card p { margin: 0; color: #52514e; font-size: .92rem; }
.badge-ok { color: #0a7a3f; font-weight: 600; }
.badge-todo { color: #9a6a00; font-weight: 600; }
div[data-testid="stMetricValue"] { font-size: 1.5rem; }
</style>
"""


def apply_style() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def header(step: int | str, title: str, subtitle: str) -> None:
    kicker = f"Step {step}" if isinstance(step, int) else step
    st.markdown(f'<div class="step-header"><div class="kicker">{kicker}</div>'
                f"<h1>{title}</h1><p>{subtitle}</p></div>", unsafe_allow_html=True)


def card(title: str, text: str) -> None:
    st.markdown(f'<div class="card"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- state
DEFAULTS = {
    "synthetic_cfg": SyntheticConfig,
    "model_cfg": ModelConfig,
    "train_cfg": TrainConfig,
    "analysis_cfg": AnalysisConfig,
    "images": dict,            # name -> SEMImage (after cropping / calibration)
    "raw_images": dict,        # name -> SEMImage as loaded
    "results": dict,           # name -> ImageResult
    "data_dir": lambda: str(DATA_DIR / "synthetic"),
    "model_dir": lambda: "",
    "evaluation": lambda: None,
}


def init_state() -> None:
    for key, factory in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = factory()


@st.cache_resource(show_spinner="Loading network …")
def _load_model_cached(path: str, mtime: float):
    from aunw.model import load_model
    return load_model(path)


def active_model():
    """The model chosen in the sidebar (or None)."""
    from aunw.training import model_file
    d = st.session_state.get("model_dir")
    if not d:
        return None
    f = model_file(Path(d))
    if not f.exists():
        return None
    return _load_model_cached(str(f), f.stat().st_mtime)


def available_models() -> list[Path]:
    from aunw.training import list_models
    return list_models(MODELS_DIR)


def dataset_ready(data_dir: str | None = None) -> bool:
    d = Path(data_dir or st.session_state["data_dir"])
    return (d / "train.npz").exists() and (d / "val.npz").exists()


@st.cache_resource
def trainer_slot() -> dict:
    """One training job per server process; survives page changes and reruns."""
    return {"monitor": None, "thread": None}


def workflow_status() -> dict[str, bool]:
    return {
        "data": dataset_ready(),
        "network": bool(available_models()),
        "training": bool(available_models()),
        "validation": st.session_state.get("evaluation") is not None,
        "images": bool(st.session_state["images"]),
        "analysis": bool(st.session_state["results"]),
        "results": bool(st.session_state["results"]),
    }


# --------------------------------------------------------------------------- widgets
def show_figure(fig, name: str, formats=("png", "svg", "pdf"), key: str | None = None,
                width: int | str = 820) -> None:
    """Render a matplotlib figure with download buttons for each format."""
    st.pyplot(fig, width=width)
    cols = st.columns(len(formats) + 3)
    mime = {"png": "image/png", "svg": "image/svg+xml", "pdf": "application/pdf"}
    for col, fmt in zip(cols, formats):
        col.download_button(f"⬇ {fmt.upper()}", figure_bytes(fig, fmt), f"{name}.{fmt}", mime[fmt],
                            key=f"dl_{key or name}_{fmt}")


def range_slider(label: str, lo: float, hi: float, value: tuple, step: float, help: str = ""):
    return tuple(st.slider(label, lo, hi, tuple(value), step, help=help))
