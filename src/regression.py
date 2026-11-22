"""Jalur 2: K-Nearest Neighbors Regressor (dengan standardisasi fitur numerik).

Parameter ditetapkan tetap dan dipakai identik pada baseline serta seluruh
skenario Ablation Study (tidak ada tuning).
"""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline

from src.data_loader import FEATURE_COLUMNS
from src.preprocessing import build_pipeline

KNN_PARAMS = {"n_neighbors": 5, "weights": "uniform", "metric": "minkowski", "p": 2}


def build_knn() -> KNeighborsRegressor:
    return KNeighborsRegressor(**KNN_PARAMS)


def train_regressor(X_train: pd.DataFrame, y_train, features=None, scale: bool = True) -> Pipeline:
    """Melatih pipeline baru (preprocessor di-fit ulang pada training).

    scale=True  : baseline dan Eksperimen 1-6 (ada standardisasi numerik)
    scale=False : Eksperimen 7 (tanpa standardisasi; imputasi dan encoding tetap ada)
    """
    features = list(FEATURE_COLUMNS if features is None else features)
    pipeline = build_pipeline(build_knn(), features, scale=scale)
    pipeline.fit(X_train[features], y_train)
    return pipeline


def compute_regression_metrics(y_true, y_pred) -> dict:
    """MAE, RMSE, dan R-squared."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R²": float(r2_score(y_true, y_pred)),
    }


@dataclass
class RegressionResult:
    pipeline: Pipeline
    y_true: pd.Series
    y_pred: np.ndarray
    summary: dict
    comparison: pd.DataFrame


def evaluate_regressor(pipeline: Pipeline, X_test: pd.DataFrame, y_test, features=None) -> RegressionResult:
    features = list(FEATURE_COLUMNS if features is None else features)
    y_pred = pipeline.predict(X_test[features])
    comparison = pd.DataFrame({
        "Aktual (menit)": np.asarray(y_test, dtype=float),
        "Prediksi (menit)": np.round(y_pred, 2),
    }, index=pd.Series(y_test).index)
    comparison["Selisih (prediksi - aktual)"] = (comparison["Prediksi (menit)"] - comparison["Aktual (menit)"]).round(2)
    comparison["Selisih absolut"] = comparison["Selisih (prediksi - aktual)"].abs()
    return RegressionResult(pipeline, pd.Series(y_test), y_pred,
                            compute_regression_metrics(y_test, y_pred), comparison)


def model_params_table(pipeline: Pipeline) -> pd.DataFrame:
    """Seluruh parameter KNN yang digunakan."""
    params = pipeline.named_steps["model"].get_params()
    return pd.DataFrame({"Parameter": list(params.keys()), "Nilai": [str(v) for v in params.values()]})


def plot_actual_vs_predicted(result: RegressionResult):
    """Scatter aktual vs prediksi dan histogram selisih."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    actual = np.asarray(result.y_true, dtype=float)
    pred = np.asarray(result.y_pred, dtype=float)
    axes[0].scatter(actual, pred, alpha=0.6, color="#1F4E79")
    lim = [min(actual.min(), pred.min()), max(actual.max(), pred.max())]
    axes[0].plot(lim, lim, "--", color="#C0392B", label="Prediksi = Aktual")
    axes[0].set_xlabel("Waktu pengiriman aktual (menit)")
    axes[0].set_ylabel("Waktu pengiriman prediksi (menit)")
    axes[0].set_title("Aktual vs Prediksi (data testing)")
    axes[0].legend()
    axes[1].hist(pred - actual, bins=25, color="#8FB3D9", edgecolor="white")
    axes[1].axvline(0, color="#C0392B", linestyle="--")
    axes[1].set_xlabel("Selisih (prediksi - aktual), menit")
    axes[1].set_ylabel("Jumlah pesanan")
    axes[1].set_title("Distribusi selisih prediksi")
    fig.tight_layout()
    return fig
