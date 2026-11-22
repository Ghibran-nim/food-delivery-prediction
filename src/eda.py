"""Exploratory Data Analysis (deskriptif).

EDA hanya menggambarkan data; tidak ada parameter model yang dipelajari di sini
dan tidak ada kesimpulan sebab-akibat yang ditarik dari grafik atau korelasi.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data_loader import CATEGORICAL_FEATURES, CLASS_ORDER, NUMERIC_FEATURES, TARGET_COL

BLUE = "#1F4E79"
LIGHT_BLUE = "#8FB3D9"

CATEGORY_ORDERS = {
    "Weather": ["Clear", "Foggy", "Rainy", "Snowy", "Windy"],
    "Traffic_Level": ["Low", "Medium", "High"],
    "Time_of_Day": ["Morning", "Afternoon", "Evening", "Night"],
    "Vehicle_Type": ["Bike", "Scooter", "Car"],
}


def _order(df: pd.DataFrame, col: str) -> list[str]:
    present = list(df[col].dropna().unique())
    known = [v for v in CATEGORY_ORDERS.get(col, []) if v in present]
    return known + sorted(v for v in present if v not in known)


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Statistik deskriptif kolom numerik (fitur numerik dan target)."""
    return df[NUMERIC_FEATURES + [TARGET_COL]].describe().T.round(3)


def categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Jumlah dan persentase setiap nilai pada fitur kategorikal (missing ditampilkan)."""
    rows = []
    for col in CATEGORICAL_FEATURES:
        counts = df[col].fillna("(kosong)").value_counts()
        for value, n in counts.items():
            rows.append({"Kolom": col, "Nilai": value, "Jumlah": int(n),
                         "Persentase (%)": round(n / len(df) * 100, 2)})
    return pd.DataFrame(rows)


def numeric_correlation(df: pd.DataFrame) -> pd.DataFrame:
    return df[NUMERIC_FEATURES + [TARGET_COL]].corr().round(3)


def plot_histogram(df: pd.DataFrame, col: str, title: str, xlabel: str, bins: int = 30):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    sns.histplot(df[col].dropna(), bins=bins, kde=True, color=BLUE, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Jumlah pesanan")
    fig.tight_layout()
    return fig


def plot_countplot(df: pd.DataFrame, col: str, title: str):
    series = df[col].fillna("(kosong)")
    order = _order(df, col) + (["(kosong)"] if df[col].isna().any() else [])
    fig, ax = plt.subplots(figsize=(6.5, 4))
    sns.countplot(x=series, order=order, color=LIGHT_BLUE, ax=ax)
    for container in ax.containers:
        ax.bar_label(container, padding=2)
    ax.set_title(title)
    ax.set_xlabel(col)
    ax.set_ylabel("Jumlah pesanan")
    fig.tight_layout()
    return fig


def plot_distance_vs_delivery(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sns.scatterplot(data=df, x="Distance_km", y=TARGET_COL, alpha=0.6, color=BLUE, ax=ax)
    ax.set_title("Jarak pengiriman vs waktu pengiriman")
    ax.set_xlabel("Jarak pengiriman (km)")
    ax.set_ylabel("Waktu pengiriman (menit)")
    fig.tight_layout()
    return fig


def plot_box_by_category(df: pd.DataFrame, col: str, title: str):
    data = df.dropna(subset=[col])
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sns.boxplot(data=data, x=col, y=TARGET_COL, order=_order(df, col), color=LIGHT_BLUE, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(col)
    ax.set_ylabel("Waktu pengiriman (menit)")
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4.8))
    sns.heatmap(numeric_correlation(df), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Korelasi antarvariabel numerik")
    fig.tight_layout()
    return fig


def plot_class_distribution(table: pd.DataFrame):
    """Persentase kelas pada data penuh, training, dan testing."""
    long = pd.melt(
        table, id_vars="Kelas",
        value_vars=["Data penuh (%)", "Training (%)", "Testing (%)"],
        var_name="Bagian data", value_name="Persentase (%)",
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=long, x="Kelas", y="Persentase (%)", hue="Bagian data", order=CLASS_ORDER, ax=ax)
    ax.set_title("Distribusi kelas Fast / Medium / Slow")
    fig.tight_layout()
    return fig
