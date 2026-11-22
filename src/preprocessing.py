"""Preprocessing berbasis Pipeline dan ColumnTransformer.

Prinsip utama (sesuai laporan, Bagian II subbab 2.4):
- Imputasi, encoding, dan standardisasi HANYA di-fit pada data training.
- Data testing hanya ditransformasi dengan parameter dari training.
- Standardisasi hanya untuk jalur KNN Regressor dan hanya pada kolom numerik
  (hasil One-Hot Encoding tidak distandardisasi).
- Setiap pemanggilan `build_preprocessor` membuat objek BARU, sehingga setiap
  skenario Ablation Study di-fit ulang sesuai fitur yang tersisa.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES


def check_features(features) -> list[str]:
    """Memastikan hanya fitur prediktor yang sah (Order_ID/target otomatis ditolak)."""
    features = list(features)
    unknown = [f for f in features if f not in FEATURE_COLUMNS]
    if unknown:
        raise ValueError(
            f"Fitur tidak dikenal: {unknown}. Hanya 7 fitur prediktor yang boleh dipakai: {FEATURE_COLUMNS}."
        )
    if not features:
        raise ValueError("Minimal harus ada satu fitur.")
    return features


def split_feature_types(features) -> tuple[list[str], list[str]]:
    """Memisahkan fitur yang tersisa menjadi (numerik, kategorikal)."""
    features = check_features(features)
    numeric = [c for c in NUMERIC_FEATURES if c in features]
    categorical = [c for c in CATEGORICAL_FEATURES if c in features]
    return numeric, categorical


def build_preprocessor(features, scale: bool) -> ColumnTransformer:
    """Membuat preprocessor baru (belum di-fit).

    Numerik    : imputer median (+ StandardScaler bila scale=True)
    Kategorikal: imputer modus -> OneHotEncoder (encoding SELALU ada)
    """
    numeric, categorical = split_feature_types(features)
    transformers = []
    if numeric:
        steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale:
            steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(steps), numeric))
    if categorical:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categorical,
        ))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def build_pipeline(model, features, scale: bool) -> Pipeline:
    """Pipeline lengkap: preprocessor + model."""
    return Pipeline([("preprocessor", build_preprocessor(features, scale)), ("model", model)])


# ---------------------------------------------------------------------------
# Fungsi bantu untuk menampilkan dan memverifikasi hasil preprocessing
# ---------------------------------------------------------------------------
def transformed_frame(preprocessor: ColumnTransformer, X: pd.DataFrame) -> pd.DataFrame:
    """Data hasil transformasi (preprocessor harus sudah di-fit) sebagai DataFrame."""
    names = list(preprocessor.get_feature_names_out())
    return pd.DataFrame(preprocessor.transform(X), columns=names, index=X.index)


def get_imputation_statistics(preprocessor: ColumnTransformer, features) -> pd.DataFrame:
    """Nilai imputasi yang dipelajari dari data training."""
    numeric, categorical = split_feature_types(features)
    rows = []
    if numeric:
        imputer = preprocessor.named_transformers_["num"].named_steps["imputer"]
        for col, value in zip(numeric, imputer.statistics_):
            rows.append({"Fitur": col, "Jenis": "Numerik", "Strategi": "Median", "Nilai (dari training)": value})
    if categorical:
        imputer = preprocessor.named_transformers_["cat"].named_steps["imputer"]
        for col, value in zip(categorical, imputer.statistics_):
            rows.append({"Fitur": col, "Jenis": "Kategorikal", "Strategi": "Modus", "Nilai (dari training)": value})
    return pd.DataFrame(rows).astype({"Nilai (dari training)": "object"})


def get_scaler_statistics(preprocessor: ColumnTransformer, features) -> pd.DataFrame | None:
    """Mean dan standar deviasi StandardScaler (None bila tidak ada scaler)."""
    numeric, _ = split_feature_types(features)
    if not numeric:
        return None
    steps = preprocessor.named_transformers_["num"].named_steps
    if "scaler" not in steps:
        return None
    scaler = steps["scaler"]
    return pd.DataFrame({"Fitur": numeric, "Mean (training)": scaler.mean_, "Std (training)": scaler.scale_})


def has_scaler(pipeline_or_preprocessor) -> bool:
    """True bila ada StandardScaler pada langkah numerik."""
    pre = pipeline_or_preprocessor
    if isinstance(pre, Pipeline):
        pre = pre.named_steps["preprocessor"]
    for name, transformer, _cols in pre.transformers:
        if name == "num" and isinstance(transformer, Pipeline) and "scaler" in transformer.named_steps:
            return True
    return False


def has_encoder(pipeline_or_preprocessor) -> bool:
    """True bila ada OneHotEncoder pada langkah kategorikal."""
    pre = pipeline_or_preprocessor
    if isinstance(pre, Pipeline):
        pre = pre.named_steps["preprocessor"]
    for name, transformer, _cols in pre.transformers:
        if name == "cat" and isinstance(transformer, Pipeline) and "encoder" in transformer.named_steps:
            return True
    return False


def missing_before_after(X_train: pd.DataFrame, X_test: pd.DataFrame, features, scale: bool) -> dict:
    """Menghitung missing values sebelum dan sesudah preprocessing.

    Preprocessor di-fit pada X_train saja, lalu X_test hanya ditransformasi.
    """
    features = check_features(features)
    pre = build_preprocessor(features, scale)
    train_out = pre.fit_transform(X_train[features])
    test_out = pre.transform(X_test[features])
    table = pd.DataFrame({
        "Fitur": features,
        "Training sebelum": X_train[features].isna().sum().values.astype(int),
        "Testing sebelum": X_test[features].isna().sum().values.astype(int),
    })
    return {
        "table": table,
        "before_train": int(X_train[features].isna().sum().sum()),
        "before_test": int(X_test[features].isna().sum().sum()),
        "after_train": int(np.isnan(train_out).sum()),
        "after_test": int(np.isnan(test_out).sum()),
        "n_columns_after": int(train_out.shape[1]),
        "preprocessor": pre,
    }
