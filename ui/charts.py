"""Grafik bertema Light/Dark (lapisan tampilan).

Seluruh data yang digambar diambil dari objek hasil yang sama dengan tabel (tidak ada
perhitungan metrik di sini). Yang berbeda dari fungsi grafik di `src/` hanyalah warna,
font, dan tata letak agar mengikuti tema aktif.
"""
from __future__ import annotations

import functools
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from src.data_loader import CLASS_ORDER, TARGET_COL
from src.eda import CATEGORY_ORDERS
from ui import theme

CLF_METRICS = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1-score (macro)"]
REG_METRICS = ["MAE", "RMSE", "R²"]
HIGHER_IS_BETTER = {
    "Accuracy": True, "Precision (macro)": True, "Recall (macro)": True,
    "F1-score (macro)": True, "R²": True, "MAE": False, "RMSE": False,
}


def themed(fn):
    """Seluruh pembuatan figure dijalankan di dalam konteks tema agar teks dan sumbu ikut tema."""
    @functools.wraps(fn)
    def wrapper(*args, mode: str = "light", **kwargs):
        mode = theme.resolve_theme(mode)
        with plt.rc_context(theme.chart_rc(mode)):
            return fn(*args, mode=mode, t=theme.tokens(mode), **kwargs)
    return wrapper


def _order(df: pd.DataFrame, col: str) -> list:
    present = list(df[col].dropna().unique())
    known = [v for v in CATEGORY_ORDERS.get(col, []) if v in present]
    return known + sorted(v for v in present if v not in known)


def _sequential_cmap(t: dict) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("fd_seq", [theme.blend(t["primary"], t["card"], 0.06), t["primary"]])


def _diverging_cmap(t: dict) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("fd_div", [t["danger"], t["card"], t["primary"]])


def _fix_annotation_colors(ax, t: dict) -> None:
    """Memilih warna angka tiap sel heatmap: hitam atau putih, mana yang kontrasnya lebih tinggi terhadap warna sel.

    Untuk warna sel apa pun, salah satu dari hitam/putih selalu memberi kontras minimal sekitar 4.58:1 (>= WCAG AA)."""
    from matplotlib.colors import to_hex

    mesh = ax.collections[0]
    mesh.update_scalarmappable()
    face = mesh.get_facecolor()
    candidates = ["#000000", "#FFFFFF"]
    for text_artist, rgba in zip(ax.texts, face):
        cell = to_hex(rgba)
        text_artist.set_color(max(candidates, key=lambda col: theme.contrast_ratio(col, cell)))


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------
@themed
def histogram(df, col, title, xlabel, bins=30, *, mode, t):
    fig, ax = plt.subplots(figsize=(6.5, 3.9))
    sns.histplot(df[col].dropna(), bins=bins, kde=True, color=t["primary"], edgecolor=t["card"], ax=ax)
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Jumlah pesanan")
    fig.tight_layout()
    return fig


@themed
def countplot(df, col, title, *, mode, t):
    series = df[col].fillna("(kosong)")
    order = _order(df, col) + (["(kosong)"] if df[col].isna().any() else [])
    fig, ax = plt.subplots(figsize=(6.5, 3.9))
    sns.countplot(x=series, order=order, color=t["primary"], ax=ax)
    for container in ax.containers:
        ax.bar_label(container, padding=2, color=t["text"])
    ax.set_title(title, loc="left")
    ax.set_xlabel(col)
    ax.set_ylabel("Jumlah pesanan")
    fig.tight_layout()
    return fig


@themed
def distance_vs_delivery(df, *, mode, t):
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    sns.scatterplot(data=df, x="Distance_km", y=TARGET_COL, alpha=0.6, color=t["primary"], edgecolor=t["card"], ax=ax)
    ax.set_title("Jarak pengiriman vs waktu pengiriman", loc="left")
    ax.set_xlabel("Jarak pengiriman (km)")
    ax.set_ylabel("Waktu pengiriman (menit)")
    fig.tight_layout()
    return fig


@themed
def box_by_category(df, col, title, *, mode, t):
    data = df.dropna(subset=[col])
    fill = theme.blend(t["primary"], t["card"], 0.35)
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    sns.boxplot(
        data=data, x=col, y=TARGET_COL, order=_order(df, col), ax=ax,
        boxprops={"facecolor": fill, "edgecolor": t["text"]},
        medianprops={"color": t["text"]}, whiskerprops={"color": t["text"]},
        capprops={"color": t["text"]}, flierprops={"markeredgecolor": t["muted"], "markerfacecolor": "none"},
    )
    ax.set_title(title, loc="left")
    ax.set_xlabel(col)
    ax.set_ylabel("Waktu pengiriman (menit)")
    fig.tight_layout()
    return fig


@themed
def correlation_heatmap(corr: pd.DataFrame, *, mode, t):
    fig, ax = plt.subplots(figsize=(6, 4.6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=_diverging_cmap(t), vmin=-1, vmax=1,
                linewidths=1, linecolor=t["card"], ax=ax, annot_kws={"size": 10})
    ax.set_title("Korelasi antarvariabel numerik", loc="left")
    ax.grid(False)
    _fix_annotation_colors(ax, t)
    fig.tight_layout()
    return fig


@themed
def class_distribution(table, *, mode, t):
    long = pd.melt(table, id_vars="Kelas", value_vars=["Data penuh (%)", "Training (%)", "Testing (%)"],
                   var_name="Bagian data", value_name="Persentase (%)")
    palette = {"Data penuh (%)": t["primary"], "Training (%)": t["secondary"], "Testing (%)": t["muted"]}
    fig, ax = plt.subplots(figsize=(7, 3.9))
    sns.barplot(data=long, x="Kelas", y="Persentase (%)", hue="Bagian data", order=CLASS_ORDER,
                palette=palette, ax=ax)
    ax.set_title("Distribusi kelas Fast / Medium / Slow", loc="left")
    ax.legend(title=None, frameon=True)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------------------------
def _draw_confusion(ax, confusion: pd.DataFrame, t: dict, title: str, compact: bool):
    data = confusion.copy()
    data.index = [i.replace("Aktual: ", "") for i in data.index]
    data.columns = [c.replace("Prediksi: ", "") for c in data.columns]
    sns.heatmap(data, annot=True, fmt="d", cmap=_sequential_cmap(t), cbar=False, linewidths=1.5,
                linecolor=t["card"], ax=ax, annot_kws={"size": 11 if compact else 13, "weight": "600"})
    ax.set_title(title, loc="left", fontsize=10 if compact else 11)
    ax.set_xlabel("Kelas prediksi")
    ax.set_ylabel("Kelas aktual")
    ax.grid(False)
    ax.tick_params(axis="y", rotation=0)
    _fix_annotation_colors(ax, t)


@themed
def confusion_matrix_figure(confusion, *, mode, t):
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    _draw_confusion(ax, confusion, t, "Confusion Matrix (data testing)", compact=False)
    fig.tight_layout()
    return fig


@themed
def confusion_grid(confusions: dict, clf_table: pd.DataFrame, *, mode, t):
    names = list(confusions)
    n_cols = 4
    n_rows = math.ceil(len(names) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.8 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    accuracy = clf_table.set_index("Skenario")["Accuracy"]
    for ax, name in zip(axes, names):
        _draw_confusion(ax, confusions[name], t, f"{name}\nAccuracy = {accuracy[name]:.3f}", compact=True)
    for ax in axes[len(names):]:
        ax.axis("off")
    fig.suptitle("Confusion Matrix per skenario - Decision Tree (data testing)", fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ---------------------------------------------------------------------------
# Regresi dan Ablation
# ---------------------------------------------------------------------------
@themed
def actual_vs_predicted(y_true, y_pred, *, mode, t):
    actual = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    axes[0].scatter(actual, pred, alpha=0.65, color=t["primary"], edgecolor=t["card"], linewidth=0.5)
    lim = [min(actual.min(), pred.min()), max(actual.max(), pred.max())]
    axes[0].plot(lim, lim, "--", color=t["danger"], label="Prediksi = Aktual")
    axes[0].set_xlabel("Waktu pengiriman aktual (menit)")
    axes[0].set_ylabel("Waktu pengiriman prediksi (menit)")
    axes[0].set_title("Aktual vs Prediksi (data testing)", loc="left")
    axes[0].legend()
    axes[1].hist(pred - actual, bins=25, color=t["secondary"], edgecolor=t["card"])
    axes[1].axvline(0, color=t["danger"], linestyle="--")
    axes[1].set_xlabel("Selisih (prediksi - aktual), menit")
    axes[1].set_ylabel("Jumlah pesanan")
    axes[1].set_title("Distribusi selisih prediksi", loc="left")
    fig.tight_layout()
    return fig


def _short(name: str) -> str:
    return name.replace("Eksperimen ", "Eks. ")


def _bar_panel(ax, table: pd.DataFrame, metric: str, t: dict):
    labels = [_short(n) for n in table["Skenario"]]
    values = table[metric].values
    colors = [t["primary"]] + [theme.blend(t["primary"], t["card"], 0.45)] * (len(values) - 1)
    bars = ax.bar(labels, values, color=colors)
    ax.axhline(values[0], color=t["danger"], linestyle="--", linewidth=1, label="Baseline")
    direction = "lebih tinggi lebih baik" if HIGHER_IS_BETTER[metric] else "lebih rendah lebih baik"
    ax.set_title(f"{metric} ({direction})", loc="left", fontsize=10)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2, color=t["text"])
    ax.margins(y=0.16)


@themed
def ablation_classification(clf_table, *, mode, t):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, metric in zip(axes.ravel(), CLF_METRICS):
        _bar_panel(ax, clf_table, metric, t)
    fig.suptitle("Ablation Study - Decision Tree Classifier vs Baseline", fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


@themed
def ablation_regression(reg_table, *, mode, t):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, metric in zip(axes, REG_METRICS):
        _bar_panel(ax, reg_table, metric, t)
    fig.suptitle("Ablation Study - KNN Regressor vs Baseline", fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig
