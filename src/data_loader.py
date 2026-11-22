"""Pemuatan dan validasi dataset, pembentukan target, serta pembagian data.

Modul ini juga menyimpan semua konstanta penelitian (nama kolom, batas
kategori, rasio split, random state) agar tidak tersebar di banyak file.
Semua nilai mengikuti laporan Tugas Kecerdasan Artifisial.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Konstanta penelitian (sesuai laporan, Bagian II subbab 2.2 dan 2.3)
# ---------------------------------------------------------------------------
ID_COL = "Order_ID"
TARGET_COL = "Delivery_Time_min"
CATEGORY_COL = "Delivery_Category"

NUMERIC_FEATURES = ["Distance_km", "Preparation_Time_min", "Courier_Experience_yrs"]
CATEGORICAL_FEATURES = ["Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES  # 7 fitur prediktor
REQUIRED_COLUMNS = [ID_COL] + FEATURE_COLUMNS + [TARGET_COL]

CLASS_ORDER = ["Fast", "Medium", "Slow"]
FAST_MAX = 30     # Fast   : <= 30 menit
MEDIUM_MAX = 45   # Medium : > 30 sampai 45 menit, Slow: > 45 menit

TEST_SIZE = 0.2   # 80% training : 20% testing
RANDOM_STATE = 42

COLUMN_FUNCTIONS = {
    "Order_ID": "Identitas pesanan (tidak digunakan sebagai fitur)",
    "Distance_km": "Jarak pengiriman (fitur numerik)",
    "Weather": "Kondisi cuaca (fitur kategorikal)",
    "Traffic_Level": "Tingkat kemacetan (fitur kategorikal)",
    "Time_of_Day": "Waktu pemesanan/pengiriman (fitur kategorikal)",
    "Vehicle_Type": "Jenis kendaraan (fitur kategorikal)",
    "Preparation_Time_min": "Waktu persiapan makanan (fitur numerik)",
    "Courier_Experience_yrs": "Pengalaman kurir dalam tahun (fitur numerik)",
    "Delivery_Time_min": "Target klasifikasi dan regresi",
}


class DatasetError(Exception):
    """Error dengan pesan yang aman ditampilkan langsung ke pengguna."""


# ---------------------------------------------------------------------------
# Membaca dan memvalidasi dataset
# ---------------------------------------------------------------------------
def load_csv(source) -> pd.DataFrame:
    """Membaca CSV dari path atau file yang diunggah."""
    try:
        df = pd.read_csv(source)
    except pd.errors.EmptyDataError as exc:
        raise DatasetError("File CSV kosong.") from exc
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DatasetError(f"File tidak dapat dibaca sebagai CSV: {exc}") from exc
    if df.empty:
        raise DatasetError("File CSV tidak berisi baris data.")
    return df


def validate_columns(df: pd.DataFrame) -> None:
    """Memastikan kolom wajib ada dan kolom numerik bertipe numerik."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DatasetError(
            "Kolom wajib tidak ditemukan: " + ", ".join(missing)
            + ". Kolom pada file: " + ", ".join(map(str, df.columns)) + "."
        )
    not_numeric = [
        c for c in NUMERIC_FEATURES + [TARGET_COL]
        if not pd.api.types.is_numeric_dtype(df[c])
    ]
    if not_numeric:
        raise DatasetError(
            "Kolom berikut seharusnya numerik tetapi bertipe lain: "
            + ", ".join(f"{c} ({df[c].dtype})" for c in not_numeric) + "."
        )


def column_check(df: pd.DataFrame) -> pd.DataFrame:
    """Tabel kecocokan kolom dataset dengan kolom pada laporan."""
    return pd.DataFrame({
        "Kolom": list(COLUMN_FUNCTIONS.keys()),
        "Fungsi": list(COLUMN_FUNCTIONS.values()),
        "Ada di dataset": ["Ya" if c in df.columns else "TIDAK ADA" for c in COLUMN_FUNCTIONS],
    })


def dataset_overview(df: pd.DataFrame) -> dict:
    """Ringkasan struktur: ukuran, tipe data, dan data duplikat."""
    dtypes = pd.DataFrame({
        "Kolom": list(df.columns),
        "Tipe data": [str(t) for t in df.dtypes],
        "Terisi (non-null)": df.notna().sum().values,
    })
    without_id = [c for c in df.columns if c != ID_COL]
    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "dtypes": dtypes,
        "n_duplicate_rows": int(df.duplicated().sum()),
        # Order_ID yang unik bisa menyembunyikan duplikat; ini pengecekan tambahan
        "n_duplicate_ignoring_id": int(df.duplicated(subset=without_id).sum()),
        "n_duplicate_order_id": int(df[ID_COL].duplicated().sum()) if ID_COL in df.columns else None,
    }


def missing_report(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """Jumlah dan persentase missing values per kolom."""
    columns = list(df.columns if columns is None else columns)
    counts = df[columns].isna().sum()
    return pd.DataFrame({
        "Kolom": columns,
        "Missing": counts.values.astype(int),
        "Persentase (%)": (counts.values / max(len(df), 1) * 100).round(2),
    })


def order_id_report(df: pd.DataFrame) -> dict:
    """Evaluasi terpisah untuk Order_ID (bukan fitur prediktor)."""
    n_missing = int(df[ID_COL].isna().sum())
    n_dup = int(df[ID_COL].dropna().duplicated().sum())
    if n_missing == 0 and n_dup == 0:
        note = "Order_ID lengkap dan unik. Kolom ini tidak digunakan sebagai fitur."
    else:
        note = (
            "Order_ID memiliki masalah, tetapi kolom ini tidak digunakan sebagai fitur "
            "sehingga tidak memengaruhi model. Baris tidak dikeluarkan hanya karena Order_ID bermasalah."
        )
    return {"n_missing": n_missing, "n_duplicated": n_dup, "note": note}


def target_report(df: pd.DataFrame) -> dict:
    """Pemeriksaan nilai minimum, maksimum, dan desimal pada target."""
    s = df[TARGET_COL].dropna()
    return {
        "n_missing": int(df[TARGET_COL].isna().sum()),
        "min": float(s.min()),
        "max": float(s.max()),
        "has_decimal": bool((s % 1 != 0).any()),
        "dtype": str(df[TARGET_COL].dtype),
    }


def outlier_report(df: pd.DataFrame, columns) -> pd.DataFrame:
    """Ringkasan outlier metode IQR. Bersifat DESKRIPTIF: tidak ada data yang dihapus."""
    rows = []
    for col in columns:
        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        rows.append({
            "Kolom": col,
            "Q1": round(float(q1), 3),
            "Q3": round(float(q3), 3),
            "Batas bawah": round(float(low), 3),
            "Batas atas": round(float(high), 3),
            "Jumlah di luar batas": int(((s < low) | (s > high)).sum()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pembentukan target klasifikasi
# ---------------------------------------------------------------------------
def make_delivery_category(delivery_time) -> pd.Series:
    """Fast (<= 30), Medium (> 30 sampai 45), Slow (> 45).

    Memakai batas interval (30, 45] sehingga nilai desimal pun tertangani.
    Missing value tidak diizinkan (target tidak boleh diimputasi).
    """
    values = pd.Series(delivery_time)
    if values.isna().any():
        raise ValueError("Delivery_Time_min tidak boleh berisi missing value saat membentuk kategori.")
    labels = pd.cut(
        values, bins=[-np.inf, FAST_MAX, MEDIUM_MAX, np.inf], labels=CLASS_ORDER, right=True
    )
    return pd.Series(labels.astype(str), index=values.index, name=CATEGORY_COL)


def class_distribution(labels) -> pd.DataFrame:
    """Jumlah dan persentase setiap kelas."""
    counts = pd.Series(labels).value_counts().reindex(CLASS_ORDER, fill_value=0)
    total = int(counts.sum())
    pct = (counts.values / total * 100).round(2) if total else np.zeros(len(CLASS_ORDER))
    return pd.DataFrame({
        "Kelas": CLASS_ORDER,
        "Jumlah": counts.values.astype(int),
        "Persentase (%)": pct,
    })


# ---------------------------------------------------------------------------
# Pembagian data (satu kali, dipakai oleh semua model dan semua eksperimen)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SplitInfo:
    train_idx: np.ndarray
    test_idx: np.ndarray
    stratified: bool
    note: str
    test_size: float
    random_state: int

    def fingerprint(self) -> str:
        """Sidik jari indeks split; nilainya harus identik pada semua pengujian."""
        h = hashlib.sha256()
        h.update(np.asarray(self.train_idx, dtype=np.int64).tobytes())
        h.update(np.asarray(self.test_idx, dtype=np.int64).tobytes())
        return h.hexdigest()[:12]


def make_split(labels, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE) -> SplitInfo:
    """Split 80:20 stratified berdasarkan kategori. Bila tidak memungkinkan,
    gunakan split acak biasa dan jelaskan alasannya pada `note`."""
    labels = pd.Series(labels).reset_index(drop=True)
    idx = np.arange(len(labels))
    try:
        train_idx, test_idx = train_test_split(
            idx, test_size=test_size, random_state=random_state, stratify=labels
        )
        return SplitInfo(np.sort(train_idx), np.sort(test_idx), True,
                         "Stratified split berhasil; proporsi kelas terjaga.", test_size, random_state)
    except ValueError as exc:
        train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=random_state)
        note = (
            f"Stratified split tidak dapat dilakukan ({exc}). Solusi: split acak biasa dengan "
            "random_state yang sama; proporsi kelas pada testing mungkin tidak persis sama."
        )
        return SplitInfo(np.sort(train_idx), np.sort(test_idx), False, note, test_size, random_state)


@dataclass
class ExperimentData:
    """Semua data yang dibutuhkan model, dengan satu split yang sama."""
    df: pd.DataFrame            # data siap supervised (baris target kosong sudah dikeluarkan)
    X: pd.DataFrame             # 7 fitur prediktor saja
    y_class: pd.Series          # kategori Fast/Medium/Slow
    y_reg: pd.Series            # Delivery_Time_min (numerik)
    split: SplitInfo
    n_input_rows: int
    n_removed_missing_target: int
    removal_reason: str

    @property
    def X_train(self): return self.X.iloc[self.split.train_idx]
    @property
    def X_test(self): return self.X.iloc[self.split.test_idx]
    @property
    def y_class_train(self): return self.y_class.iloc[self.split.train_idx]
    @property
    def y_class_test(self): return self.y_class.iloc[self.split.test_idx]
    @property
    def y_reg_train(self): return self.y_reg.iloc[self.split.train_idx]
    @property
    def y_reg_test(self): return self.y_reg.iloc[self.split.test_idx]


def build_experiment_data(df: pd.DataFrame, test_size: float = TEST_SIZE,
                          random_state: int = RANDOM_STATE) -> ExperimentData:
    """Validasi -> keluarkan baris target kosong -> bentuk label -> split sekali."""
    validate_columns(df)
    n_input = len(df)
    missing_target = df[TARGET_COL].isna()
    n_removed = int(missing_target.sum())
    data = df.loc[~missing_target].reset_index(drop=True)
    if len(data) < 20:
        raise DatasetError("Data terlalu sedikit untuk dibagi menjadi training dan testing.")

    reason = (
        "Delivery_Time_min adalah target. Target tidak boleh diimputasi, sehingga baris yang "
        "targetnya kosong dikeluarkan dari pelatihan supervised learning."
        if n_removed else "Tidak ada baris yang dikeluarkan: target tidak memiliki missing values."
    )
    y_reg = data[TARGET_COL].copy()
    y_class = make_delivery_category(y_reg)
    X = data[FEATURE_COLUMNS].copy()   # Order_ID dan target tidak ikut sebagai fitur
    split = make_split(y_class, test_size, random_state)
    return ExperimentData(data, X, y_class, y_reg, split, n_input, n_removed, reason)


def class_distribution_all(data: ExperimentData) -> pd.DataFrame:
    """Distribusi kelas pada data penuh, training, dan testing dalam satu tabel."""
    full = class_distribution(data.y_class)
    tr = class_distribution(data.y_class_train)
    te = class_distribution(data.y_class_test)
    return pd.DataFrame({
        "Kelas": CLASS_ORDER,
        "Data penuh (n)": full["Jumlah"], "Data penuh (%)": full["Persentase (%)"],
        "Training (n)": tr["Jumlah"], "Training (%)": tr["Persentase (%)"],
        "Testing (n)": te["Jumlah"], "Testing (%)": te["Persentase (%)"],
    })


def imbalance_note(distribution: pd.DataFrame) -> str:
    """Ringkasan keseimbangan kelas pada data penuh. Tidak mengubah batas kategori."""
    top = distribution.loc[distribution["Jumlah"].idxmax()]
    nonzero = distribution[distribution["Jumlah"] > 0]
    bottom = nonzero.loc[nonzero["Jumlah"].idxmin()]
    ratio = top["Jumlah"] / bottom["Jumlah"]
    return (
        f"Kelas terbanyak adalah {top['Kelas']} ({top['Persentase (%)']:.2f}%), kelas tersedikit "
        f"adalah {bottom['Kelas']} ({bottom['Persentase (%)']:.2f}%); rasio sekitar {ratio:.1f}:1. "
        f"Menebak selalu '{top['Kelas']}' sudah menghasilkan Accuracy sekitar "
        f"{top['Persentase (%)']:.1f}%, sehingga Accuracy harus dibaca bersama Precision, Recall, dan "
        "F1-score per kelas. Batas kategori TIDAK diubah otomatis; perubahan hanya dilakukan "
        "setelah didokumentasikan dan disetujui."
    )
