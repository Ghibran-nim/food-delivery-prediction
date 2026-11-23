"""Pembungkus Streamlit tipis di atas `ui.markup` dan `ui.theme` (lapisan tampilan saja)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from ui import markup, theme


def _html(content: str) -> None:
    st.markdown(content, unsafe_allow_html=True)


def inject_theme(mode: str) -> None:
    """Menyuntikkan CSS tema aktif. Dipanggil pada setiap rerun sebelum konten dirender."""
    _html(f"<style>{theme.build_css(mode)}</style>")


def page_header(title: str, subtitle: str | None = None) -> None:
    _html(markup.header_html(title, subtitle))


def section(title: str, description: str | None = None) -> None:
    _html(markup.section_html(title, description))


def metric_cards(items: list[dict]) -> None:
    _html(markup.cards_html(items))


def kv_panel(title: str, rows: list[tuple]) -> None:
    _html(markup.panel_kv_html(title, rows))


def callout(kind: str, text: str, title: str | None = None, allow_html: bool = False) -> None:
    _html(markup.callout_html(kind, text, title, allow_html))


def show_table(df: pd.DataFrame, **kwargs) -> None:
    _html(markup.table_html(df, **kwargs))


def show_fig(fig) -> None:
    st.pyplot(fig)
    plt.close(fig)


def brand() -> None:
    _html(markup.brand_html())


def side_label(text: str) -> None:
    _html(markup.side_label_html(text))


def badge(text: str, kind: str = "neutral") -> str:
    return markup.badge_html(text, kind)


def result_card(label: str, value: str, hint: str | None = None, badge_html: str | None = None) -> None:
    _html(markup.result_html(label, value, hint, badge_html))


def html(content: str) -> None:
    _html(content)
