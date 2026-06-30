from pathlib import Path
import runpy
import sys

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / "kelompok3_harga_pangan"

if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))


st.set_page_config(
    page_title="ChiliForecast",
    page_icon=":hot_pepper:",
    layout="wide",
)


PAGE_FILES = {
    "Dashboard": "kelompok3_harga_pangan/pages/1_Dashboard.py",
    "EDA": "kelompok3_harga_pangan/pages/2_EDA.py",
    "Prediksi": "kelompok3_harga_pangan/pages/3_Prediksi.py",
    "Tentang Model": "kelompok3_harga_pangan/pages/4_Tentang_Model.py",
}


def run_legacy_navigation():
    selected_page = st.sidebar.radio(
        "Navigasi",
        list(PAGE_FILES.keys()),
    )

    runpy.run_path(str(ROOT_DIR / PAGE_FILES[selected_page]), run_name="__main__")


if hasattr(st, "navigation") and hasattr(st, "Page"):
    pages = [
        *[
            st.Page(page_file, title=page_name)
            for page_name, page_file in PAGE_FILES.items()
        ],
    ]

    current_page = st.navigation(pages)
    current_page.run()
else:
    run_legacy_navigation()
