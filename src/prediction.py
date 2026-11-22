"""Pelatihan model baseline dan prediksi pesanan baru.

Prediksi memakai pipeline yang sama persis dengan yang dilatih pada data
training (imputasi, encoding, dan standardisasi KNN dibawa oleh pipeline),
sehingga tidak ada preprocessing manual yang berbeda.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.classification import ClassificationResult, evaluate_classifier, train_classifier
from src.data_loader import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, ExperimentData
from src.regression import RegressionResult, evaluate_regressor, train_regressor

DISCLAIMER = (
    "Hasil ini merupakan estimasi berdasarkan pola pada dataset yang digunakan, bukan jaminan "
    "waktu pengiriman aktual. Kedua model dilatih terpisah, sehingga kategori dan estimasi menit "
    "tidak selalu selaras."
)


@dataclass
class BaselineModels:
    clf: ClassificationResult
    reg: RegressionResult


def train_baseline_models(data: ExperimentData) -> BaselineModels:
    """Melatih dan mengevaluasi kedua model dengan seluruh 7 fitur (baseline)."""
    clf_pipe = train_classifier(data.X_train, data.y_class_train, FEATURE_COLUMNS)
    reg_pipe = train_regressor(data.X_train, data.y_reg_train, FEATURE_COLUMNS, scale=True)
    return BaselineModels(
        clf=evaluate_classifier(clf_pipe, data.X_test, data.y_class_test, FEATURE_COLUMNS),
        reg=evaluate_regressor(reg_pipe, data.X_test, data.y_reg_test, FEATURE_COLUMNS),
    )


def build_order_frame(order: dict) -> pd.DataFrame:
    """Mengubah input satu pesanan menjadi DataFrame 1 baris dengan urutan kolom yang benar."""
    missing = [c for c in FEATURE_COLUMNS if c not in order]
    extra = [c for c in order if c not in FEATURE_COLUMNS]
    if missing or extra:
        raise ValueError(f"Input harus berisi tepat 7 fitur prediktor. Kurang: {missing}. Berlebih: {extra}.")
    return pd.DataFrame([{c: order[c] for c in FEATURE_COLUMNS}])[FEATURE_COLUMNS]


def predict_new_order(models: BaselineModels, order: dict) -> dict:
    """Prediksi kategori (Decision Tree) dan estimasi menit (KNN) untuk satu pesanan."""
    frame = build_order_frame(order)
    return {
        "kategori": str(models.clf.pipeline.predict(frame)[0]),
        "menit": float(models.reg.pipeline.predict(frame)[0]),
    }


def get_form_options(X_train: pd.DataFrame) -> dict:
    """Pilihan kategori dan rentang numerik dari data training untuk formulir."""
    return {
        "categories": {c: sorted(X_train[c].dropna().unique().tolist()) for c in CATEGORICAL_FEATURES},
        "ranges": {c: (float(X_train[c].min()), float(X_train[c].max())) for c in NUMERIC_FEATURES},
    }


def range_warnings(order: dict, ranges: dict) -> list[str]:
    """Peringatan bila nilai numerik berada di luar rentang data training."""
    warnings = []
    for col, (low, high) in ranges.items():
        value = order[col]
        if value < low or value > high:
            warnings.append(
                f"{col} = {value} berada di luar rentang data training ({low:g} - {high:g}); "
                "estimasi pada nilai di luar rentang kurang dapat diandalkan."
            )
    return warnings
