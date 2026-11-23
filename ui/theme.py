"""Sistem desain (lapisan tampilan saja): palet Light/Dark, CSS global, dan utilitas kontras.

Modul ini tidak mengimpor Streamlit sehingga dapat diuji tersendiri.
Tidak ada kode Machine Learning di sini.
"""
from __future__ import annotations

from string import Template

FONT_STACK = "'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"

# Warna utama mengikuti spesifikasi desain. Token turunan (hover, table_head, shadow, on_primary)
# hanya menambah komponen yang tidak ditentukan di spesifikasi.
PALETTES = {
    "light": {
        "bg": "#F8FAFC", "sidebar": "#FFFFFF", "card": "#FFFFFF",
        "primary": "#2563EB", "secondary": "#0D9488",
        "text": "#0F172A", "muted": "#64748B", "border": "#E2E8F0",
        "success": "#16A34A", "warning": "#D97706", "danger": "#DC2626",
        "on_primary": "#FFFFFF", "hover": "#F1F5F9", "table_head": "#F1F5F9",
        "shadow": "0 1px 2px rgba(15,23,42,0.05), 0 1px 3px rgba(15,23,42,0.06)",
    },
    "dark": {
        "bg": "#0B1120", "sidebar": "#111827", "card": "#172033",
        "primary": "#60A5FA", "secondary": "#2DD4BF",
        "text": "#F1F5F9", "muted": "#94A3B8", "border": "#334155",
        "success": "#4ADE80", "warning": "#FBBF24", "danger": "#F87171",
        "on_primary": "#0B1120", "hover": "#1E2A41", "table_head": "#1B2740",
        "shadow": "0 1px 2px rgba(0,0,0,0.35)",
    },
}


def resolve_theme(value) -> str:
    """Mengembalikan 'dark' atau 'light' (nilai tidak dikenal dianggap 'light')."""
    return "dark" if str(value).strip().lower() == "dark" else "light"


# ---------------------------------------------------------------------------
# Utilitas warna dan kontras (WCAG)
# ---------------------------------------------------------------------------
def _rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _hex(rgb) -> str:
    return "#%02X%02X%02X" % tuple(int(round(x)) for x in rgb)


def blend(fg: str, bg: str, alpha: float) -> str:
    """Warna fg dengan opasitas alpha di atas bg (hasil komposit, tanpa transparansi)."""
    f, b = _rgb(fg), _rgb(bg)
    return _hex([alpha * f[i] + (1 - alpha) * b[i] for i in range(3)])


def _linear(channel: int) -> float:
    c = channel / 255
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def luminance(hex_color: str) -> float:
    r, g, b = _rgb(hex_color)
    return 0.2126 * _linear(r) + 0.7152 * _linear(g) + 0.0722 * _linear(b)


def contrast_ratio(a: str, b: str) -> float:
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def tokens(mode: str) -> dict:
    """Seluruh token warna untuk satu tema, termasuk warna tint hasil komposit."""
    mode = resolve_theme(mode)
    t = dict(PALETTES[mode])
    t["scheme"] = mode
    t["font"] = FONT_STACK
    for name in ("primary", "secondary", "success", "warning", "danger"):
        t[f"tint_{name}"] = blend(t[name], t["card"], 0.12)
    t["primary_border"] = blend(t["primary"], t["card"], 0.45)
    return t


# ---------------------------------------------------------------------------
# Pengaturan grafik Matplotlib mengikuti tema
# ---------------------------------------------------------------------------
def pick_font() -> str:
    """Font grafik: Inter bila terpasang, selain itu font sans-serif yang tersedia."""
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Inter", "Plus Jakarta Sans", "Segoe UI", "DejaVu Sans"):
        if name in available:
            return name
    return "sans-serif"


def chart_rc(mode: str) -> dict:
    t = tokens(mode)
    return {
        "figure.facecolor": t["card"], "axes.facecolor": t["card"], "savefig.facecolor": t["card"],
        "axes.edgecolor": t["border"], "axes.labelcolor": t["text"], "axes.titlecolor": t["text"],
        "text.color": t["text"], "xtick.color": t["text"], "ytick.color": t["text"],
        "axes.grid": True, "axes.axisbelow": True, "grid.color": t["border"], "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.facecolor": t["card"], "legend.edgecolor": t["border"], "legend.labelcolor": t["text"],
        "font.family": pick_font(), "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "600",
        "axes.labelsize": 10,
    }


# ---------------------------------------------------------------------------
# CSS global. Memakai ${token} yang diganti saat build agar komponen mengikuti tema aktif.
# ---------------------------------------------------------------------------
CSS_TEMPLATE = Template(r"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root { color-scheme: ${scheme}; }

/* ===== Kerangka aplikasi ===== */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
  background: ${bg}; color: ${text}; font-family: ${font};
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stDecoration"], footer { display: none; }
.stApp .block-container { max-width: 1240px; padding: 2rem 2rem 4rem 2rem; }

/* ===== Sidebar ===== */
[data-testid="stSidebar"] { background: ${sidebar}; border-right: 1px solid ${border}; }
[data-testid="stSidebar"] > div:first-child { background: ${sidebar}; }
[data-testid="stSidebar"] [role="radiogroup"] { gap: 0.15rem; }
[data-testid="stSidebar"] [role="radiogroup"] label {
  width: 100%; padding: 0.5rem 0.7rem; border-radius: 9px; border: 1px solid transparent; cursor: pointer;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: ${hover}; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  background: ${tint_primary}; border-color: ${primary_border};
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p { color: ${text}; font-weight: 600; }
[data-testid="stSidebar"] [role="radiogroup"] label p { font-size: 0.92rem; }

/* ===== Teks dasar ===== */
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
  color: ${text}; font-family: ${font}; letter-spacing: -0.01em;
}
.stApp p, .stApp li, .stApp label, .stApp [data-testid="stMarkdownContainer"] {
  color: ${text}; font-family: ${font};
}
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] p { color: ${muted}; }
.stApp a { color: ${primary}; }
.stApp code { background: ${hover}; color: ${text}; border-radius: 6px; }
.stApp [data-testid="stCode"] pre, .stApp pre {
  background: ${table_head}; border: 1px solid ${border}; border-radius: 10px;
}
.stApp [data-testid="stCode"] code, .stApp [data-testid="stCode"] code span, .stApp pre code { color: ${text}; background: transparent; }

/* ===== Tombol ===== */
.stApp [data-testid="stBaseButton-primary"], .stApp [data-testid="stBaseButton-primaryFormSubmit"] {
  background: ${primary}; border: 1px solid ${primary}; border-radius: 9px; box-shadow: none;
}
.stApp [data-testid="stBaseButton-primary"] p, .stApp [data-testid="stBaseButton-primaryFormSubmit"] p {
  color: ${on_primary}; font-weight: 600;
}
.stApp [data-testid="stBaseButton-primary"]:hover, .stApp [data-testid="stBaseButton-primaryFormSubmit"]:hover {
  filter: brightness(1.08); border-color: ${primary};
}
.stApp [data-testid="stBaseButton-secondary"], .stApp [data-testid="stBaseButton-secondaryFormSubmit"] {
  background: ${card}; border: 1px solid ${border}; border-radius: 9px; box-shadow: none;
}
.stApp [data-testid="stBaseButton-secondary"] p, .stApp [data-testid="stBaseButton-secondaryFormSubmit"] p {
  color: ${text}; font-weight: 500;
}
.stApp [data-testid="stBaseButton-secondary"]:hover, .stApp [data-testid="stBaseButton-secondaryFormSubmit"]:hover {
  background: ${hover}; border-color: ${primary};
}

/* ===== Input ===== */
.stApp [data-baseweb="input"], .stApp [data-baseweb="base-input"], .stApp [data-baseweb="select"] > div {
  background: ${card}; border-color: ${border}; border-radius: 9px;
}
.stApp [data-baseweb="input"] input, .stApp [data-baseweb="select"] div, .stApp [data-baseweb="select"] input {
  color: ${text}; -webkit-text-fill-color: ${text};
}
.stApp [data-testid="stNumberInput"] button { background: ${card}; color: ${text}; border-color: ${border}; }
[data-baseweb="popover"] [data-baseweb="menu"], [data-baseweb="popover"] ul {
  background: ${card}; border: 1px solid ${border};
}
[data-baseweb="popover"] li { color: ${text}; background: ${card}; }
[data-baseweb="popover"] li:hover, [data-baseweb="popover"] li[aria-selected="true"] { background: ${hover}; }
.stApp [data-testid="stFileUploaderDropzone"] {
  background: ${card}; border: 1.5px dashed ${border}; border-radius: 12px;
}
.stApp [data-testid="stFileUploaderDropzone"] span, .stApp [data-testid="stFileUploaderDropzone"] small { color: ${text}; }

/* ===== Tab, expander, form ===== */
.stApp [data-baseweb="tab-list"] { gap: 0.25rem; }
.stApp button[data-baseweb="tab"] { background: transparent; }
.stApp button[data-baseweb="tab"] p { color: ${text}; font-weight: 500; }
.stApp button[data-baseweb="tab"][aria-selected="true"] p { color: ${primary}; font-weight: 600; }
.stApp [data-baseweb="tab-highlight"] { background: ${primary}; }
.stApp [data-baseweb="tab-border"] { background: ${border}; }
.stApp [data-testid="stExpander"] { background: ${card}; border: 1px solid ${border}; border-radius: 12px; }
.stApp [data-testid="stExpander"] summary, .stApp [data-testid="stExpander"] summary p { color: ${text}; }
.stApp [data-testid="stForm"] { background: ${card}; border: 1px solid ${border}; border-radius: 12px; padding: 1.25rem; }

/* ===== Komponen kustom ===== */
.stApp .fd-brand { padding: 0.25rem 0.25rem 1rem 0.25rem; border-bottom: 1px solid ${border}; margin-bottom: 0.9rem; }
.stApp .fd-brand-title { font-size: 0.92rem; font-weight: 700; letter-spacing: 0.07em; color: ${text}; }
.stApp .fd-brand-sub { font-size: 0.78rem; color: ${muted}; margin-top: 0.2rem; }
.stApp .fd-side-label { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.06em; color: ${muted}; margin: 1rem 0 0.35rem 0.25rem; text-transform: uppercase; }
.stApp .fd-header { margin-bottom: 1.1rem; }
.stApp .fd-title { font-size: 1.75rem; font-weight: 700; color: ${text}; line-height: 1.2; letter-spacing: -0.02em; }
.stApp .fd-subtitle { font-size: 0.95rem; color: ${muted}; margin-top: 0.3rem; }
.stApp .fd-section { margin: 1.7rem 0 0.5rem 0; }
.stApp .fd-section-title { font-size: 1.1rem; font-weight: 600; color: ${text}; }
.stApp .fd-section-desc { font-size: 0.85rem; color: ${muted}; margin-top: 0.15rem; }

.stApp .fd-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 0.85rem; margin: 0.5rem 0 1rem 0; }
.stApp .fd-card { background: ${card}; border: 1px solid ${border}; border-radius: 12px; padding: 0.95rem 1.1rem; box-shadow: ${shadow}; }
.stApp .fd-card-accent { border-top: 3px solid ${primary}; }
.stApp .fd-card-label { font-size: 0.78rem; font-weight: 500; color: ${muted}; }
.stApp .fd-card-value { font-size: 1.6rem; font-weight: 700; color: ${text}; margin-top: 0.2rem; font-variant-numeric: tabular-nums; }
.stApp .fd-card-hint { font-size: 0.76rem; color: ${muted}; margin-top: 0.2rem; }

.stApp .fd-panel { background: ${card}; border: 1px solid ${border}; border-radius: 12px; padding: 1rem 1.15rem; box-shadow: ${shadow}; margin-bottom: 0.85rem; }
.stApp .fd-panel-title { font-size: 0.95rem; font-weight: 600; color: ${text}; margin-bottom: 0.45rem; }
.stApp .fd-text { font-size: 0.86rem; line-height: 1.5; color: ${text}; margin: 0.35rem 0; }
.stApp .fd-text-muted { color: ${muted}; }
.stApp .fd-panel-tag { margin-top: 0.55rem; }
.stApp .fd-list { margin: 0.35rem 0 0 1.15rem; padding: 0; }
.stApp .fd-list li { margin: 0.3rem 0; line-height: 1.5; color: ${text}; }
.stApp .fd-kv { display: flex; justify-content: space-between; gap: 1rem; padding: 0.38rem 0; border-bottom: 1px solid ${border}; font-size: 0.85rem; }
.stApp .fd-kv:last-child { border-bottom: none; }
.stApp .fd-kv-key { color: ${muted}; }
.stApp .fd-kv-val { color: ${text}; font-weight: 500; text-align: right; word-break: break-word; }

.stApp .fd-table-wrap { overflow: auto; border: 1px solid ${border}; border-radius: 12px; background: ${card}; margin: 0.4rem 0 1rem 0; box-shadow: ${shadow}; }
.stApp table.fd-table { border-collapse: collapse; width: 100%; font-size: 0.84rem; color: ${text}; }
.stApp table.fd-table th { position: sticky; top: 0; background: ${table_head}; color: ${text}; font-weight: 600; text-align: left; padding: 0.55rem 0.8rem; border-bottom: 1px solid ${border}; white-space: nowrap; }
.stApp table.fd-table td { padding: 0.5rem 0.8rem; border-bottom: 1px solid ${border}; color: ${text}; white-space: nowrap; }
.stApp table.fd-table tr:last-child td { border-bottom: none; }
.stApp table.fd-table tbody tr:hover td { background: ${hover}; }
.stApp table.fd-table th.num, .stApp table.fd-table td.num { text-align: right; font-variant-numeric: tabular-nums; }
.stApp table.fd-table td.fd-na { color: ${muted}; font-style: italic; }
.stApp .fd-arrow-good { color: ${success}; font-size: 0.7rem; margin-right: 0.3rem; }
.stApp .fd-arrow-bad { color: ${danger}; font-size: 0.7rem; margin-right: 0.3rem; }
.stApp .fd-arrow-flat { color: ${muted}; font-size: 0.7rem; margin-right: 0.3rem; }

.stApp .fd-callout { border: 1px solid ${border}; border-left: 4px solid ${primary}; background: ${tint_primary}; border-radius: 10px; padding: 0.7rem 1rem; margin: 0.5rem 0 1rem 0; font-size: 0.88rem; color: ${text}; }
.stApp .fd-callout.success { border-left-color: ${success}; background: ${tint_success}; }
.stApp .fd-callout.warning { border-left-color: ${warning}; background: ${tint_warning}; }
.stApp .fd-callout.danger { border-left-color: ${danger}; background: ${tint_danger}; }
.stApp .fd-callout-title { font-weight: 600; margin-bottom: 0.15rem; }

.stApp .fd-badge { display: inline-block; padding: 0.15rem 0.65rem; border-radius: 999px; font-size: 0.75rem; font-weight: 600; border: 1px solid ${border}; color: ${text}; background: ${hover}; }
.stApp .fd-badge.success { background: ${tint_success}; border-color: ${success}; }
.stApp .fd-badge.warning { background: ${tint_warning}; border-color: ${warning}; }
.stApp .fd-badge.danger { background: ${tint_danger}; border-color: ${danger}; }
.stApp .fd-badge.primary { background: ${tint_primary}; border-color: ${primary}; }

.stApp .fd-result { background: ${card}; border: 1px solid ${border}; border-top: 3px solid ${primary}; border-radius: 12px; padding: 1.1rem 1.3rem; box-shadow: ${shadow}; }
.stApp .fd-result-label { font-size: 0.8rem; color: ${muted}; font-weight: 500; margin-bottom: 0.45rem; }
.stApp .fd-result-value { font-size: 2.1rem; font-weight: 700; color: ${text}; font-variant-numeric: tabular-nums; line-height: 1.15; }
.stApp .fd-result-hint { font-size: 0.78rem; color: ${muted}; margin-top: 0.4rem; }

/* ===== Responsif ===== */
@media (max-width: 768px) {
  .stApp .block-container { padding: 1rem 1rem 3rem 1rem; }
  .stApp .fd-title { font-size: 1.4rem; }
  .stApp .fd-cards { grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); }
  .stApp .fd-card-value { font-size: 1.3rem; }
  .stApp .fd-result-value { font-size: 1.7rem; }
}
""")


def build_css(mode: str) -> str:
    """CSS lengkap untuk tema yang dipilih."""
    return CSS_TEMPLATE.substitute(tokens(mode))
