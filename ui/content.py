"""Teks penjelasan untuk pengguna (bahasa Indonesia, lapisan tampilan saja).

Tidak ada perhitungan model di sini. Contoh pada penjelasan metrik dibentuk dari nilai metrik
yang sudah dihitung (bila tersedia); bila belum ada dataset, contoh diberi label 'Misalnya'
dan memakai angka ilustrasi.
"""
from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal

import pandas as pd

from src import data_loader as dl


def fmt_id(value: float, decimals: int) -> str:
    """Format angka gaya Indonesia (koma desimal) dengan pembulatan konvensional (setengah ke atas)."""
    quantum = Decimal(1).scaleb(-decimals)
    rounded = Decimal(repr(round(float(value), 6))).quantize(quantum, rounding=ROUND_HALF_UP)
    return f"{rounded:.{decimals}f}".replace(".", ",")


# ---------------------------------------------------------------------------
# Halaman Prediction
# ---------------------------------------------------------------------------
PREDICTION_NOTE_TITLE = "Cara membaca dua hasil prediksi"


def category_rule_text() -> str:
    return (f"Batas kategori: Fast \u2264 {dl.FAST_MAX} menit, Medium > {dl.FAST_MAX} sampai {dl.MEDIUM_MAX} menit, "
            f"Slow > {dl.MEDIUM_MAX} menit.")


def prediction_note_items() -> list[tuple[str, str]]:
    return [
        ("Dua model, dua tujuan.",
         "Decision Tree mengelompokkan waktu pengiriman ke dalam kategori (Fast, Medium, atau Slow), sedangkan "
         "KNN memperkirakan waktu pengiriman sebagai angka dalam menit. Keduanya bekerja sendiri-sendiri."),
        ("Hasilnya tidak selalu terlihat selaras.",
         "Misalnya, Decision Tree menghasilkan Slow, sedangkan estimasi KNN berupa angka menit yang mungkin tidak "
         "sepenuhnya sesuai dengan batas kategori. Ini bukan otomatis berarti aplikasi salah, karena kedua model "
         "dilatih dan memprediksi dengan cara yang berbeda. " + category_rule_text()),
        ("Perkiraan, bukan jaminan.",
         "Hasil didasarkan pada pola data pelatihan dan bukan jaminan waktu pengiriman yang sebenarnya."),
    ]


def consistency_note(category: str, minutes: float) -> str | None:
    """Catatan bila kategori Decision Tree berbeda dari kategori menurut estimasi menit KNN.

    Hanya membandingkan dua hasil yang sudah ada memakai batas kategori yang sama; tidak mengubah hasil apa pun.
    """
    if not math.isfinite(minutes):
        return None
    implied = str(dl.make_delivery_category(pd.Series([minutes])).iloc[0])
    if implied == category:
        return None
    return (f"Pada hasil ini, Decision Tree menghasilkan {category}, sedangkan estimasi KNN "
            f"({fmt_id(minutes, 1)} menit) berada pada rentang kategori {implied}. Ini dapat terjadi karena kedua "
            "model bekerja secara terpisah dan bukan otomatis menunjukkan kesalahan aplikasi.")


# ---------------------------------------------------------------------------
# Dashboard: Memahami Metrik Evaluasi
# ---------------------------------------------------------------------------
SECTION_TITLE = "Memahami Metrik Evaluasi"
SECTION_DESC = "Panduan singkat membaca angka kinerja model di atas, dikelompokkan menurut jenis tugasnya."
EXTRA_CAPTION = ("Precision, Recall, dan F1-score (macro) menilai setiap kelas secara terpisah; "
                 "rinciannya ada di halaman Model Classification.")
LOWER_BETTER = "Semakin kecil, semakin baik"
HIGHER_BETTER = "Semakin tinggi, semakin baik"

# Angka ilustrasi (hanya dipakai bila belum ada dataset dan selalu diberi label 'Misalnya')
_DEMO = {"Accuracy": 0.845, "MAE": 8.74, "RMSE": 12.0, "R²": 0.69}


def metric_groups(clf_summary: dict | None = None, reg_summary: dict | None = None) -> list[dict]:
    """Penjelasan empat metrik, dikelompokkan: Classification (Accuracy) dan Regression (MAE, RMSE, R²)."""
    prefix_c = "Pada hasil saat ini:" if clf_summary else "Misalnya:"
    prefix_r = "Pada hasil saat ini:" if reg_summary else "Misalnya:"
    acc = (clf_summary or _DEMO)["Accuracy"]
    mae = (reg_summary or _DEMO)["MAE"]
    rmse = (reg_summary or _DEMO)["RMSE"]
    r2 = (reg_summary or _DEMO)["R²"]

    acc_pct = fmt_id(acc * 100, 1)
    if r2 >= 0:
        r2_example = (f"{prefix_r} R² {fmt_id(r2, 2)} berarti model dapat menjelaskan sekitar "
                      f"{fmt_id(r2 * 100, 0)}% variasi waktu pengiriman pada data pengujian.")
    else:
        r2_example = (f"{prefix_r} R² {fmt_id(r2, 2)} bernilai negatif, artinya model lebih buruk daripada "
                      "sekadar memakai rata-rata waktu pengiriman sebagai tebakan.")

    return [
        {"task": "Classification - Decision Tree", "metrics": [
            {"title": "Accuracy", "direction": HIGHER_BETTER,
             "definition": "Persentase prediksi kategori yang benar dibandingkan seluruh data pengujian.",
             "example": f"{prefix_c} Accuracy {acc_pct}% berarti model berhasil memprediksi kategori dengan benar "
                        f"pada sekitar {acc_pct}% data pengujian."},
        ]},
        {"task": "Regression - KNN Regressor", "metrics": [
            {"title": "MAE (Mean Absolute Error)", "direction": LOWER_BETTER,
             "definition": "Rata-rata selisih absolut antara waktu pengiriman hasil prediksi dan waktu pengiriman aktual.",
             "example": f"{prefix_r} MAE {fmt_id(mae, 2)} menit berarti rata-rata prediksi model berbeda sekitar "
                        f"{fmt_id(mae, 2)} menit dari waktu aktual."},
            {"title": "RMSE (Root Mean Squared Error)", "direction": LOWER_BETTER,
             "definition": "Ukuran besarnya kesalahan prediksi yang memberi penalti lebih besar pada kesalahan yang besar.",
             "example": f"{prefix_r} RMSE {fmt_id(rmse, 2)} menit adalah besarnya kesalahan prediksi yang sudah "
                        "memperhitungkan penalti lebih besar untuk kesalahan ekstrem."},
            {"title": "R² (R-Squared)", "direction": HIGHER_BETTER + " (mendekati 1)",
             "definition": "Seberapa baik model menjelaskan variasi waktu pengiriman pada data pengujian. "
                           "R² bukan persentase prediksi yang benar.",
             "example": r2_example},
        ]},
    ]
