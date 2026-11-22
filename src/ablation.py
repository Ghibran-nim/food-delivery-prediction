"""Ablation Study (Bagian IV laporan).

Total 15 pengujian:
- Decision Tree Classifier : Baseline + Eksperimen 1-6      = 7 skenario
- KNN Regressor            : Baseline + Eksperimen 1-7      = 8 skenario

Aturan yang dijaga oleh kode ini:
- Semua skenario memakai indeks training-testing yang SAMA (dari ExperimentData).
- Parameter model SAMA di semua skenario (DT_PARAMS dan KNN_PARAMS, tanpa tuning).
- Setiap skenario membuat pipeline BARU sehingga imputasi, encoding, dan
  standardisasi di-fit ulang pada data training sesuai fitur yang tersisa.
- Encoding kategorikal selalu ada. Eksperimen 7 hanya menghilangkan scaler.
- Semua angka berasal dari eksekusi model, tidak ada angka manual.
"""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd

from src.classification import DT_PARAMS, compute_classification_metrics, train_classifier
from src.data_loader import ExperimentData, FEATURE_COLUMNS
from src.regression import KNN_PARAMS, compute_regression_metrics, train_regressor

CLF_NAME = "Decision Tree Classifier"
REG_NAME = "KNN Regressor"
CLF_METRICS = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1-score (macro)"]
REG_METRICS = ["MAE", "RMSE", "R²"]
HIGHER_IS_BETTER = {
    "Accuracy": True, "Precision (macro)": True, "Recall (macro)": True,
    "F1-score (macro)": True, "R²": True, "MAE": False, "RMSE": False,
}
EXPECTED_CLF_RUNS = 7
EXPECTED_REG_RUNS = 8


@dataclass(frozen=True)
class Scenario:
    key: str
    name: str
    removed: str
    features: tuple
    scale: bool          # hanya berlaku untuk KNN; Decision Tree tidak pernah distandardisasi
    models: tuple


def define_scenarios() -> list[Scenario]:
    """Delapan skenario sesuai Bagian 4.3 laporan."""
    all_features = tuple(FEATURE_COLUMNS)

    def without(*cols):
        return tuple(c for c in all_features if c not in cols)

    both = (CLF_NAME, REG_NAME)
    return [
        Scenario("baseline", "Baseline", "- (seluruh fitur dan komponen digunakan)", all_features, True, both),
        Scenario("exp1", "Eksperimen 1", "Distance_km", without("Distance_km"), True, both),
        Scenario("exp2", "Eksperimen 2", "Weather, Traffic_Level", without("Weather", "Traffic_Level"), True, both),
        Scenario("exp3", "Eksperimen 3", "Time_of_Day", without("Time_of_Day"), True, both),
        Scenario("exp4", "Eksperimen 4", "Vehicle_Type", without("Vehicle_Type"), True, both),
        Scenario("exp5", "Eksperimen 5", "Preparation_Time_min", without("Preparation_Time_min"), True, both),
        Scenario("exp6", "Eksperimen 6", "Courier_Experience_yrs", without("Courier_Experience_yrs"), True, both),
        Scenario("exp7", "Eksperimen 7", "Standardisasi fitur numerik", all_features, False, (REG_NAME,)),
    ]


def scenario_table() -> pd.DataFrame:
    """Tabel rancangan skenario (untuk ditampilkan sebelum eksperimen dijalankan)."""
    rows = []
    for sc in define_scenarios():
        rows.append({
            "Skenario": sc.name,
            "Dihilangkan": sc.removed,
            "Fitur yang digunakan": ", ".join(sc.features),
            "Model": " dan ".join(sc.models) if len(sc.models) > 1 else sc.models[0] + " saja",
        })
    return pd.DataFrame(rows)


@dataclass
class AblationResult:
    clf_table: pd.DataFrame
    reg_table: pd.DataFrame
    details: pd.DataFrame
    pipelines: dict
    n_runs: int
    split_fingerprint: str
    n_test: int


def _add_deltas(rows: list[dict], metrics: list[str]) -> pd.DataFrame:
    table = pd.DataFrame(rows)
    baseline = table.iloc[0]
    if baseline["Skenario"] != "Baseline":
        raise RuntimeError("Baris pertama tabel harus Baseline.")
    for metric in metrics:
        table[f"Δ {metric}"] = table[metric] - baseline[metric]
    return table


def run_ablation(data: ExperimentData, progress=None) -> AblationResult:
    """Menjalankan seluruh 15 pengujian dan mengembalikan tabel hasil.

    progress: fungsi opsional progress(selesai, total, teks) untuk menampilkan kemajuan.
    """
    scenarios = define_scenarios()
    total = sum(len(sc.models) for sc in scenarios)
    fingerprint = data.split.fingerprint()
    clf_rows, reg_rows, details, pipelines = [], [], [], {}
    done = 0

    for sc in scenarios:
        features = list(sc.features)
        n_removed = len(FEATURE_COLUMNS) - len(features)

        if CLF_NAME in sc.models:
            pipe = train_classifier(data.X_train, data.y_class_train, features)   # tanpa scaler
            y_pred = pipe.predict(data.X_test[features])
            summary = compute_classification_metrics(data.y_class_test, y_pred)["summary"]
            clf_rows.append({"Skenario": sc.name, "Komponen yang dihilangkan": sc.removed,
                             "Jumlah fitur": len(features), **summary})
            pipelines[(sc.key, CLF_NAME)] = pipe
            details.append(_detail_row(sc, CLF_NAME, features, pipe, False, DT_PARAMS, data, fingerprint))
            done += 1
            if progress:
                progress(done, total, f"{sc.name} - {CLF_NAME}")

        if REG_NAME in sc.models:
            pipe = train_regressor(data.X_train, data.y_reg_train, features, scale=sc.scale)
            y_pred = pipe.predict(data.X_test[features])
            summary = compute_regression_metrics(data.y_reg_test, y_pred)
            reg_rows.append({"Skenario": sc.name, "Komponen yang dihilangkan": sc.removed,
                             "Jumlah fitur": len(features), **summary})
            pipelines[(sc.key, REG_NAME)] = pipe
            details.append(_detail_row(sc, REG_NAME, features, pipe, sc.scale, KNN_PARAMS, data, fingerprint))
            done += 1
            if progress:
                progress(done, total, f"{sc.name} - {REG_NAME}")

    if len(clf_rows) != EXPECTED_CLF_RUNS or len(reg_rows) != EXPECTED_REG_RUNS:
        raise RuntimeError(
            f"Jumlah pengujian tidak sesuai laporan: klasifikasi {len(clf_rows)} (seharusnya "
            f"{EXPECTED_CLF_RUNS}), regresi {len(reg_rows)} (seharusnya {EXPECTED_REG_RUNS})."
        )

    return AblationResult(
        clf_table=_add_deltas(clf_rows, CLF_METRICS),
        reg_table=_add_deltas(reg_rows, REG_METRICS),
        details=pd.DataFrame(details),
        pipelines=pipelines,
        n_runs=len(clf_rows) + len(reg_rows),
        split_fingerprint=fingerprint,
        n_test=len(data.split.test_idx),
    )


def _detail_row(sc, model_name, features, pipe, scaled, params, data, fingerprint) -> dict:
    n_cols = len(pipe.named_steps["preprocessor"].get_feature_names_out())
    return {
        "Skenario": sc.name,
        "Model": model_name,
        "Fitur yang digunakan": ", ".join(features),
        "Jumlah fitur": len(features),
        "Jumlah kolom setelah encoding": n_cols,
        "Standardisasi numerik": "Ya" if scaled else "Tidak",
        "Parameter model": ", ".join(f"{k}={v}" for k, v in params.items()),
        "Jumlah data training": len(data.split.train_idx),
        "Jumlah data testing": len(data.split.test_idx),
        "Sidik jari split": fingerprint,
    }


# ---------------------------------------------------------------------------
# Visualisasi
# ---------------------------------------------------------------------------
def _short(name: str) -> str:
    return name.replace("Eksperimen ", "Eks. ")


def _bar_panel(ax, table: pd.DataFrame, metric: str):
    labels = [_short(n) for n in table["Skenario"]]
    values = table[metric].values
    colors = ["#1F4E79"] + ["#8FB3D9"] * (len(values) - 1)
    bars = ax.bar(labels, values, color=colors)
    ax.axhline(values[0], color="#C0392B", linestyle="--", linewidth=1, label="Baseline")
    direction = "lebih tinggi lebih baik" if HIGHER_IS_BETTER[metric] else "lebih rendah lebih baik"
    ax.set_title(f"{metric} ({direction})", fontsize=10)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    ax.margins(y=0.15)


def plot_classification_comparison(clf_table: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, metric in zip(axes.ravel(), CLF_METRICS):
        _bar_panel(ax, clf_table, metric)
    fig.suptitle("Ablation Study - Decision Tree Classifier vs Baseline", fontsize=12)
    fig.tight_layout()
    return fig


def plot_regression_comparison(reg_table: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, metric in zip(axes, REG_METRICS):
        _bar_panel(ax, reg_table, metric)
    fig.suptitle("Ablation Study - KNN Regressor vs Baseline", fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Interpretasi objektif (hanya menjelaskan perubahan metrik, tanpa sebab-akibat)
# ---------------------------------------------------------------------------
def describe_change(metric: str, base: float, value: float) -> str:
    delta = value - base
    if abs(delta) < 1e-12:
        return f"{metric} tidak berubah ({base:.4f})"
    arah = "meningkat" if delta > 0 else "menurun"
    lebih_baik = (delta > 0) == HIGHER_IS_BETTER[metric]
    kesan = "performa lebih baik dari baseline" if lebih_baik else "performa lebih buruk dari baseline"
    return f"{metric} {arah} dari {base:.4f} menjadi {value:.4f} (selisih {delta:+.4f}; {kesan})"


NOTES = [
    "Hasil ini berasal dari satu pembagian training-testing yang sama untuk semua skenario, sehingga "
    "selisih yang kecil dapat dipengaruhi oleh variasi data dan belum tentu bermakna.",
    "Perubahan metrik menunjukkan keterkaitan pada eksperimen ini, bukan hubungan sebab-akibat.",
    "Hasil hanya menggambarkan pola pada dataset yang digunakan.",
]


def interpret_classification(table: pd.DataFrame, n_test: int) -> list[str]:
    base = table.iloc[0]
    lines = []
    for _, row in table.iloc[1:].iterrows():
        parts = [describe_change(m, base[m], row[m]) for m in ("Accuracy", "F1-score (macro)")]
        lines.append(f"**{row['Skenario']}** (tanpa {row['Komponen yang dihilangkan']}): " + "; ".join(parts) + ".")
    drops = table.iloc[1:].sort_values("Δ Accuracy")
    worst = drops.iloc[0]
    if worst["Δ Accuracy"] < 0:
        lines.append(f"Penurunan Accuracy terbesar terjadi pada {worst['Skenario']} "
                     f"(tanpa {worst['Komponen yang dihilangkan']}), yaitu {worst['Δ Accuracy']:+.4f}.")
    else:
        lines.append("Tidak ada skenario yang menurunkan Accuracy dibanding baseline.")
    worst_f1 = table.iloc[1:].sort_values("Δ F1-score (macro)").iloc[0]
    if worst_f1["Δ F1-score (macro)"] < 0:
        lines.append(f"Penurunan F1-score (macro) terbesar terjadi pada {worst_f1['Skenario']} "
                     f"(tanpa {worst_f1['Komponen yang dihilangkan']}), yaitu {worst_f1['Δ F1-score (macro)']:+.4f}.")
    else:
        lines.append("Tidak ada skenario yang menurunkan F1-score (macro) dibanding baseline.")
    lines.append(f"Sebagai skala: satu sampel testing setara {100 / n_test:.2f} poin persentase Accuracy "
                 f"(jumlah data testing = {n_test}).")
    return lines + NOTES


def interpret_regression(table: pd.DataFrame) -> list[str]:
    base = table.iloc[0]
    lines = []
    for _, row in table.iloc[1:].iterrows():
        parts = [describe_change(m, base[m], row[m]) for m in REG_METRICS]
        lines.append(f"**{row['Skenario']}** (tanpa {row['Komponen yang dihilangkan']}): " + "; ".join(parts) + ".")
    worst = table.iloc[1:].sort_values("Δ MAE", ascending=False).iloc[0]
    if worst["Δ MAE"] > 0:
        lines.append(f"Kenaikan MAE terbesar terjadi pada {worst['Skenario']} "
                     f"(tanpa {worst['Komponen yang dihilangkan']}), yaitu {worst['Δ MAE']:+.4f} menit.")
    else:
        lines.append("Tidak ada skenario yang menaikkan MAE dibanding baseline.")
    exp7 = table[table["Skenario"] == "Eksperimen 7"]
    if not exp7.empty:
        lines.append("Pada Eksperimen 7 hanya standardisasi yang dihilangkan; fitur, imputasi, dan encoding "
                     "sama dengan baseline. Perubahan metrik pada baris tersebut membandingkan KNN "
                     "dengan dan tanpa standardisasi pada data yang sama.")
    return lines + NOTES
