"""Pembuat HTML murni (tanpa Streamlit) untuk komponen presentasi.

Seluruh teks dari data atau pengguna di-escape. HTML dibuat satu baris tanpa indentasi
agar tidak salah ditafsirkan sebagai blok kode oleh parser Markdown Streamlit.
Nilai yang ditampilkan tidak diubah; hanya diformat untuk dibaca.
"""
from __future__ import annotations

import html
import math

import numpy as np
import pandas as pd


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def _is_missing(value) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def format_value(value, decimals: int | None = None) -> str | None:
    """Memformat satu sel. Mengembalikan None untuk nilai kosong."""
    if _is_missing(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if decimals is None:
            text = f"{v:.6f}".rstrip("0").rstrip(".")
            return "0" if text in ("", "-0") else text
        return f"{v:.{decimals}f}"
    return str(value)


def table_html(df: pd.DataFrame, decimals: int | None = None, show_index: bool = False,
               max_height: int | None = None, delta_cols: dict | None = None) -> str:
    """Tabel HTML. delta_cols: {nama_kolom: higher_is_better} untuk kolom selisih terhadap baseline.

    Pada kolom selisih ditampilkan panah arah perubahan (naik/turun); warna panah menunjukkan apakah
    perubahan itu menuju performa lebih baik (hijau) atau lebih buruk (merah) menurut arah metriknya.
    """
    delta_cols = delta_cols or {}
    columns = list(df.columns)
    numeric = {c: pd.api.types.is_numeric_dtype(df[c]) for c in columns}

    head_cells = []
    if show_index:
        head_cells.append(f'<th>{esc(df.index.name or "")}</th>')
    for c in columns:
        head_cells.append(f'<th class="{"num" if numeric[c] else ""}">{esc(c)}</th>')

    rows = []
    for idx, (index_value, record) in enumerate(df.iterrows()):
        cells = []
        if show_index:
            cells.append(f"<td>{esc(index_value)}</td>")
        for c in columns:
            value = record[c]
            cls = "num" if numeric[c] else ""
            if c in delta_cols and not _is_missing(value):
                v = float(value)
                if abs(v) < 1e-12:
                    arrow, kind = "&#9679;", "flat"
                else:
                    arrow = "&#9650;" if v > 0 else "&#9660;"
                    kind = "good" if (v > 0) == delta_cols[c] else "bad"
                text = "0" if abs(v) < 1e-12 else f"{v:+.4f}"
                cells.append(f'<td class="{cls}"><span class="fd-arrow-{kind}">{arrow}</span>{esc(text)}</td>')
                continue
            text = format_value(value, decimals)
            if text is None:
                cells.append('<td class="fd-na">NaN</td>')
            else:
                cells.append(f'<td class="{cls}">{esc(text)}</td>')
        rows.append(f"<tr>{''.join(cells)}</tr>")

    style = f' style="max-height:{int(max_height)}px"' if max_height else ""
    return (f'<div class="fd-table-wrap"{style}><table class="fd-table"><thead><tr>{"".join(head_cells)}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def cards_html(items: list[dict]) -> str:
    """Baris kartu metrik. Item: label, value, hint (opsional), help (opsional), accent (opsional)."""
    cells = []
    for item in items:
        cls = "fd-card fd-card-accent" if item.get("accent") else "fd-card"
        title = f' title="{esc(item["help"])}"' if item.get("help") else ""
        hint = f'<div class="fd-card-hint">{esc(item["hint"])}</div>' if item.get("hint") else ""
        cells.append(
            f'<div class="{cls}"{title}><div class="fd-card-label">{esc(item["label"])}</div>'
            f'<div class="fd-card-value">{esc(item["value"])}</div>{hint}</div>'
        )
    return f'<div class="fd-cards">{"".join(cells)}</div>'


def panel_kv_html(title: str, rows: list[tuple]) -> str:
    """Panel berisi pasangan kunci-nilai (mis. informasi model)."""
    body = "".join(
        f'<div class="fd-kv"><span class="fd-kv-key">{esc(k)}</span><span class="fd-kv-val">{esc(v)}</span></div>'
        for k, v in rows
    )
    return f'<div class="fd-panel"><div class="fd-panel-title">{esc(title)}</div>{body}</div>'


def callout_html(kind: str, text: str, title: str | None = None, allow_html: bool = False) -> str:
    kind = kind if kind in ("info", "success", "warning", "danger") else "info"
    cls = "fd-callout" if kind == "info" else f"fd-callout {kind}"
    heading = f'<div class="fd-callout-title">{esc(title)}</div>' if title else ""
    body = text if allow_html else esc(text)
    return f'<div class="{cls}">{heading}<div>{body}</div></div>'


def badge_html(text: str, kind: str = "neutral") -> str:
    cls = "fd-badge" if kind == "neutral" else f"fd-badge {kind}"
    return f'<span class="{cls}">{esc(text)}</span>'


def header_html(title: str, subtitle: str | None = None) -> str:
    sub = f'<div class="fd-subtitle">{esc(subtitle)}</div>' if subtitle else ""
    return f'<div class="fd-header"><div class="fd-title">{esc(title)}</div>{sub}</div>'


def section_html(title: str, description: str | None = None) -> str:
    desc = f'<div class="fd-section-desc">{esc(description)}</div>' if description else ""
    return f'<div class="fd-section"><div class="fd-section-title">{esc(title)}</div>{desc}</div>'


def brand_html() -> str:
    return ('<div class="fd-brand"><div class="fd-brand-title">FOOD DELIVERY ANALYTICS</div>'
            '<div class="fd-brand-sub"><i>Machine Learning Prediction Platform</i></div></div>')


def side_label_html(text: str) -> str:
    return f'<div class="fd-side-label">{esc(text)}</div>'


def result_html(label: str, value: str, hint: str | None = None, badge: str | None = None) -> str:
    hint_html = f'<div class="fd-result-hint">{esc(hint)}</div>' if hint else ""
    badge_part = f"<div>{badge}</div>" if badge else ""
    return (f'<div class="fd-result"><div class="fd-result-label">{esc(label)}</div>'
            f'<div class="fd-result-value">{esc(value)}</div>{badge_part}{hint_html}</div>')


def is_finite_number(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
