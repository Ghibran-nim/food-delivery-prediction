"""Jalur 1: Decision Tree Classifier (tanpa standardisasi).

Parameter ditetapkan tetap dan dipakai identik pada baseline serta seluruh
skenario Ablation Study (tidak ada tuning).
"""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text

from src.data_loader import CLASS_ORDER, FEATURE_COLUMNS
from src.preprocessing import build_pipeline

DT_PARAMS = {"criterion": "gini", "max_depth": 5, "random_state": 42}


def build_decision_tree() -> DecisionTreeClassifier:
    return DecisionTreeClassifier(**DT_PARAMS)


def train_classifier(X_train: pd.DataFrame, y_train, features=None) -> Pipeline:
    """Melatih pipeline baru (preprocessor di-fit ulang pada training). Tanpa scaler."""
    features = list(FEATURE_COLUMNS if features is None else features)
    pipeline = build_pipeline(build_decision_tree(), features, scale=False)
    pipeline.fit(X_train[features], y_train)
    return pipeline


def compute_classification_metrics(y_true, y_pred) -> dict:
    """Accuracy serta Precision/Recall/F1 per kelas dan macro average.

    Macro average = rata-rata sederhana dari ketiga kelas, sehingga kelas kecil
    (Fast) berbobot sama dengan kelas besar (Slow). Cocok untuk data yang tidak seimbang.
    """
    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_ORDER, zero_division=0
    )
    per_class = pd.DataFrame({
        "Kelas": CLASS_ORDER, "Precision": p, "Recall": r, "F1-score": f, "Support": support.astype(int)
    })
    summary = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision (macro)": float(np.mean(p)),
        "Recall (macro)": float(np.mean(r)),
        "F1-score (macro)": float(np.mean(f)),
    }
    return {"summary": summary, "per_class": per_class}


@dataclass
class ClassificationResult:
    pipeline: Pipeline
    y_true: pd.Series
    y_pred: np.ndarray
    summary: dict
    per_class: pd.DataFrame
    confusion: pd.DataFrame
    report: str


def evaluate_classifier(pipeline: Pipeline, X_test: pd.DataFrame, y_test, features=None) -> ClassificationResult:
    features = list(FEATURE_COLUMNS if features is None else features)
    y_pred = pipeline.predict(X_test[features])
    metrics = compute_classification_metrics(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)
    confusion = pd.DataFrame(
        cm,
        index=[f"Aktual: {c}" for c in CLASS_ORDER],
        columns=[f"Prediksi: {c}" for c in CLASS_ORDER],
    )
    report = classification_report(y_test, y_pred, labels=CLASS_ORDER, digits=4, zero_division=0)
    return ClassificationResult(pipeline, pd.Series(y_test), y_pred, metrics["summary"],
                                metrics["per_class"], confusion, report)


def model_params_table(pipeline: Pipeline) -> pd.DataFrame:
    """Seluruh parameter Decision Tree yang digunakan."""
    params = pipeline.named_steps["model"].get_params()
    return pd.DataFrame({"Parameter": list(params.keys()), "Nilai": [str(v) for v in params.values()]})


def get_tree_rules(pipeline: Pipeline) -> str:
    """Aturan keputusan pohon dalam bentuk teks (mudah dibaca)."""
    names = list(pipeline.named_steps["preprocessor"].get_feature_names_out())
    return export_text(pipeline.named_steps["model"], feature_names=names)


def majority_baseline_accuracy(y_train, y_test) -> tuple[str, float]:
    """Pembanding konteks: Accuracy bila selalu menebak kelas terbanyak pada training."""
    majority = pd.Series(y_train).value_counts().idxmax()
    acc = float((pd.Series(y_test) == majority).mean())
    return majority, acc


def plot_confusion_matrix(confusion: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Confusion Matrix (data testing)")
    ax.set_xlabel("Kelas prediksi")
    ax.set_ylabel("Kelas aktual")
    fig.tight_layout()
    return fig
