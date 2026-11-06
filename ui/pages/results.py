"""Step 7: statistics, figures and export."""
import io
import zipfile
from pathlib import Path

import streamlit as st

from aunw import report

from aunw.export import (export_bundle, frames_of, per_image_table, standard_figures, to_excel,
                         wires_table)
from aunw.stats import summary_table
from ui import common

FIG_TITLES = {
    "length_histogram": "Length distribution", "width_histogram": "Width distribution",
    "length_vs_width": "Length vs. width", "length_per_image": "Length per image",
    "width_per_image": "Width per image", "width_cdf": "Width CDF",
}
COLUMNS = ["n", "mean", "sd", "sem", "ci95_low", "ci95_high", "median", "median_ci_low",
           "median_ci_high", "q1", "q3", "min", "max", "cv_percent", "geo_mean", "geo_sd",
           "best_model"]


def render() -> None:
    common.header(7, "Results & export", "Population statistics of all analysed images.")
    results = list(st.session_state["results"].values())
    if not results:
        st.warning("Analyse images in step 6 first.")
        return
    wires = wires_table(results)
    if wires.empty:
        st.warning("No wires detected.")
        return

    c = st.columns([1, 1, 2])
    has_nm = "length_nm" in wires and wires["length_nm"].notna().all()
    unit = c[0].selectbox("Unit", ["nm", "µm", "px"] if has_nm else ["px"])
    edge = c[1].toggle("Edge correction (Miles–Lantuéjoul)", True,
                       help="Long wires are more often cut by the frame; weighting complete wires "
                            "by the inverse probability of fitting inside removes this bias.")
    c[2].caption("Length statistics use **complete** wires only (both tips visible). Width uses "
                 "all wires, because the width is not affected by truncation.")

    table = summary_table(wires, unit, edge, frames_of(results))
    st.markdown("### Summary")
    st.dataframe(table[[c for c in COLUMNS if c in table]].style.format(precision=3,
                 subset=[c for c in COLUMNS if c in table and c not in ("n", "best_model")]))
    st.caption("CI = 95 % confidence interval (Student t for the mean, bootstrap for the median). "
               "geo_mean/geo_sd = log-normal parameters e^μ, e^σ. best_model by AIC.")

    st.markdown("### Figures")
    figs = standard_figures(results, unit, edge)
    tabs = st.tabs([FIG_TITLES[k] for k in figs])
    for tab, (key, fig) in zip(tabs, figs.items()):
        with tab:
            common.show_figure(fig, key)

    st.markdown("### Download")
    images = {n: i.pixels for n, i in st.session_state["images"].items()}
    model = st.session_state.get("model_dir", "").split("/")[-1]
    cfg = st.session_state["analysis_cfg"]
    c = st.columns(4)
    c[0].download_button("⬇ All wires (CSV)", wires.to_csv(index=False), "aunw_wires.csv", "text/csv")
    c[1].download_button("⬇ Summary (CSV)", table.to_csv(), "aunw_summary.csv", "text/csv")
    c[2].download_button("⬇ Workbook (Excel)", to_excel(results, cfg, unit, model), "aunw_results.xlsx",
                         "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    if c[3].button("Prepare complete ZIP"):
        st.session_state["zip"] = export_bundle(results, cfg, unit, images, model)
    if "zip" in st.session_state:
        c[3].download_button("⬇ ZIP (tables, figures, overlays)", st.session_state["zip"],
                             "aunw_results.zip", "application/zip")
    st.markdown("**Save everything to a folder** (statistics, figures, Excel workbook, per-wire maps)")
    s = st.columns([3, 1, 1])
    folder = s[0].text_input("Save folder", str(st.session_state.get("save_folder",
                                                                   report.default_save_folder())),
                             label_visibility="collapsed")
    st.session_state["save_folder"] = folder
    if s[1].button("💾 Save", type="primary"):
        with st.spinner("Writing statistics, figures and per-wire maps …"):
            bundle = zipfile.ZipFile(io.BytesIO(export_bundle(results, cfg, unit, images, model)))
            files = {f"statistics/{n}": bundle.read(n) for n in bundle.namelist()}
            files.update({f"per_wire/{k}": v for k, v in report.all_files(results, images).items()})
            st.session_state["last_saved"] = str(report.save_files(files, Path(folder)))
    if st.session_state.get("last_saved"):
        st.success(f"Saved to **{st.session_state['last_saved']}**")
        if s[2].button("📂 Open folder"):
            report.open_folder(Path(st.session_state["last_saved"]))
    with st.expander("Per-image overview"):
        st.dataframe(per_image_table(results), hide_index=True)
