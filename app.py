"""Aplikasi Streamlit - Food Delivery Analytics (Machine Learning Prediction Platform).

File ini hanya menyiapkan halaman, tema, dan navigasi. Tampilan berada di folder `ui/`
dan seluruh logika penelitian (preprocessing, model, ablation, prediksi) berada di folder `src/`.

Jalankan dengan:  streamlit run app.py
"""
import streamlit as st

from ui import components, pages


def main():
    st.set_page_config(
        page_title="Food Delivery Analytics",
        page_icon=":material/analytics:",
        layout="wide",
    )
    mode = pages.init_state()
    components.inject_theme(mode)
    selected = pages.render_sidebar()
    pages.PAGES[selected]()


if __name__ == "__main__":
    main()
