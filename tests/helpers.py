"""Fungsi bantu untuk unit test (data sintetis dengan struktur yang sama seperti dataset asli)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

REAL_CSV = ROOT / "data" / "Food_Delivery_Times.csv"


def make_synthetic_df(n: int = 400, seed: int = 0, with_missing: bool = True) -> pd.DataFrame:
    """Data sintetis: 9 kolom sama dengan dataset, ketiga kelas hadir, ada missing values."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Order_ID": np.arange(1, n + 1),
        "Distance_km": rng.uniform(0.5, 20, n).round(2),
        "Weather": rng.choice(["Clear", "Rainy", "Foggy", "Snowy", "Windy"], n),
        "Traffic_Level": rng.choice(["Low", "Medium", "High"], n),
        "Time_of_Day": rng.choice(["Morning", "Afternoon", "Evening", "Night"], n),
        "Vehicle_Type": rng.choice(["Bike", "Scooter", "Car"], n),
        "Preparation_Time_min": rng.integers(5, 30, n),
        "Courier_Experience_yrs": rng.integers(0, 10, n).astype(float),
    })
    traffic = df["Traffic_Level"].map({"Low": 0, "Medium": 5, "High": 12})
    delivery = 6 + 2.0 * df["Distance_km"] + 0.8 * df["Preparation_Time_min"] + traffic + rng.normal(0, 3, n)
    df["Delivery_Time_min"] = delivery.round().astype(int).clip(lower=5)
    if with_missing:
        for col in ["Weather", "Traffic_Level", "Time_of_Day", "Courier_Experience_yrs"]:
            idx = rng.choice(n, size=max(2, n // 20), replace=False)
            df.loc[idx, col] = np.nan
    return df
