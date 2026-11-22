"""Unit test: pemuatan data, pembentukan target, split, dan preprocessing."""
import numpy as np
import pandas as pd

from tests.helpers import REAL_CSV, make_synthetic_df  # noqa: F401  (juga mengatur sys.path)
from src import data_loader as dl
from src import preprocessing as prep


def _expect_raises(exc, func, *args):
    try:
        func(*args)
    except exc:
        return
    raise AssertionError(f"Seharusnya muncul {exc.__name__}")


# --- Dataset dan kolom ------------------------------------------------------
def test_required_columns_validation():
    df = make_synthetic_df()
    dl.validate_columns(df)  # tidak error
    _expect_raises(dl.DatasetError, dl.validate_columns, df.drop(columns=["Weather"]))


def test_numeric_columns_must_be_numeric():
    df = make_synthetic_df()
    df["Distance_km"] = df["Distance_km"].astype(str)
    _expect_raises(dl.DatasetError, dl.validate_columns, df)


def test_order_id_and_target_not_in_features():
    data = dl.build_experiment_data(make_synthetic_df())
    assert dl.ID_COL not in data.X.columns
    assert dl.TARGET_COL not in data.X.columns
    assert list(data.X.columns) == dl.FEATURE_COLUMNS and len(dl.FEATURE_COLUMNS) == 7
    _expect_raises(ValueError, prep.check_features, ["Distance_km", "Order_ID"])
    _expect_raises(ValueError, prep.check_features, ["Distance_km", "Delivery_Time_min"])


# --- Target -----------------------------------------------------------------
def test_category_thresholds():
    values = pd.Series([8, 29.9, 30, 30.5, 31, 45, 45.1, 46, 153])
    expected = ["Fast", "Fast", "Fast", "Medium", "Medium", "Medium", "Slow", "Slow", "Slow"]
    assert dl.make_delivery_category(values).tolist() == expected


def test_category_rejects_missing_target():
    _expect_raises(ValueError, dl.make_delivery_category, pd.Series([10, np.nan]))


def test_regression_target_stays_numeric_and_unchanged():
    df = make_synthetic_df()
    data = dl.build_experiment_data(df)
    assert pd.api.types.is_numeric_dtype(data.y_reg)
    assert (data.y_reg.values == df[dl.TARGET_COL].values).all()
    assert set(data.y_class.unique()) <= set(dl.CLASS_ORDER)


def test_missing_target_rows_removed_not_imputed():
    df = make_synthetic_df(n=200)
    df["Delivery_Time_min"] = df["Delivery_Time_min"].astype(float)
    df.loc[[3, 10, 50], "Delivery_Time_min"] = np.nan
    data = dl.build_experiment_data(df)
    assert data.n_removed_missing_target == 3
    assert len(data.X) == 197 and not data.y_reg.isna().any()
    assert "tidak boleh diimputasi" in data.removal_reason


def test_class_distribution_sums_to_total():
    data = dl.build_experiment_data(make_synthetic_df())
    dist = dl.class_distribution_all(data)
    assert dist["Data penuh (n)"].sum() == len(data.X)
    assert dist["Training (n)"].sum() == len(data.split.train_idx)
    assert dist["Testing (n)"].sum() == len(data.split.test_idx)


# --- Split ------------------------------------------------------------------
def test_split_80_20_stratified_and_reproducible():
    data1 = dl.build_experiment_data(make_synthetic_df())
    data2 = dl.build_experiment_data(make_synthetic_df())
    n = len(data1.X)
    assert len(data1.split.test_idx) == round(n * 0.2)
    assert len(data1.split.train_idx) + len(data1.split.test_idx) == n
    assert set(data1.split.train_idx).isdisjoint(set(data1.split.test_idx))
    assert data1.split.stratified
    assert data1.split.fingerprint() == data2.split.fingerprint()
    full = data1.y_class.value_counts(normalize=True)
    test = data1.y_class_test.value_counts(normalize=True)
    for cls in dl.CLASS_ORDER:
        assert abs(full[cls] - test[cls]) < 0.03


def test_split_fallback_when_class_too_small():
    labels = pd.Series(["Slow"] * 50 + ["Medium"] * 40 + ["Fast"])  # Fast hanya 1 sampel
    split = dl.make_split(labels)
    assert split.stratified is False and "Stratified split tidak dapat dilakukan" in split.note
    assert len(split.train_idx) + len(split.test_idx) == len(labels)


# --- Preprocessing ----------------------------------------------------------
def test_preprocessor_fit_on_train_only():
    data = dl.build_experiment_data(make_synthetic_df())
    X_train, X_test = data.X_train, data.X_test
    pre = prep.build_preprocessor(dl.FEATURE_COLUMNS, scale=True)
    pre.fit(X_train)
    stats = prep.get_imputation_statistics(pre, dl.FEATURE_COLUMNS).set_index("Fitur")["Nilai (dari training)"]
    assert np.isclose(float(stats["Distance_km"]), X_train["Distance_km"].median())
    assert np.isclose(float(stats["Courier_Experience_yrs"]), X_train["Courier_Experience_yrs"].median())
    for col in dl.CATEGORICAL_FEATURES:
        assert stats[col] == X_train[col].mode().iloc[0] or stats[col] in X_train[col].values
    # StandardScaler: mean/std berasal dari training (setelah imputasi), bukan dari testing
    scaler_stats = prep.get_scaler_statistics(pre, dl.FEATURE_COLUMNS).set_index("Fitur")
    assert np.isclose(scaler_stats.loc["Distance_km", "Mean (training)"], X_train["Distance_km"].mean())
    assert not np.isclose(scaler_stats.loc["Distance_km", "Mean (training)"], X_test["Distance_km"].mean())


def test_dt_has_no_scaler_and_knn_has_scaler():
    dt_pre = prep.build_preprocessor(dl.FEATURE_COLUMNS, scale=False)
    knn_pre = prep.build_preprocessor(dl.FEATURE_COLUMNS, scale=True)
    assert not prep.has_scaler(dt_pre) and prep.has_scaler(knn_pre)
    assert prep.has_encoder(dt_pre) and prep.has_encoder(knn_pre)


def test_scaler_only_on_numeric_not_onehot():
    data = dl.build_experiment_data(make_synthetic_df())
    pre = prep.build_preprocessor(dl.FEATURE_COLUMNS, scale=True).fit(data.X_train)
    out = prep.transformed_frame(pre, data.X_train)
    onehot_cols = [c for c in out.columns if c not in dl.NUMERIC_FEATURES]
    assert len(onehot_cols) == 5 + 3 + 4 + 3
    assert set(np.unique(out[onehot_cols].values)) <= {0.0, 1.0}      # One-Hot tidak distandardisasi
    for col in dl.NUMERIC_FEATURES:
        assert abs(out[col].mean()) < 1e-9 and abs(out[col].std(ddof=0) - 1) < 1e-9


def test_no_scaler_leaves_numeric_values_in_original_scale():
    data = dl.build_experiment_data(make_synthetic_df())
    pre = prep.build_preprocessor(dl.FEATURE_COLUMNS, scale=False).fit(data.X_train)
    out = prep.transformed_frame(pre, data.X_train)
    assert out["Distance_km"].max() > 5   # masih dalam km, bukan z-score


def test_no_nan_after_preprocessing_and_missing_report():
    data = dl.build_experiment_data(make_synthetic_df())
    for scale in (False, True):
        report = prep.missing_before_after(data.X_train, data.X_test, dl.FEATURE_COLUMNS, scale)
        assert report["before_train"] > 0 and report["before_test"] >= 0
        assert report["after_train"] == 0 and report["after_test"] == 0
        assert report["n_columns_after"] == 18


def test_unknown_category_in_test_does_not_fail():
    data = dl.build_experiment_data(make_synthetic_df())
    pre = prep.build_preprocessor(dl.FEATURE_COLUMNS, scale=True).fit(data.X_train)
    new = data.X_test.iloc[:3].copy()
    new["Weather"] = "Hail"   # kategori yang tidak pernah muncul di training
    out = pre.transform(new)
    assert out.shape == (3, 18) and not np.isnan(out).any()


def test_preprocessor_respects_remaining_features():
    pre = prep.build_preprocessor(["Weather", "Traffic_Level", "Distance_km"], scale=True)
    data = dl.build_experiment_data(make_synthetic_df())
    pre.fit(data.X_train[["Weather", "Traffic_Level", "Distance_km"]])
    names = list(pre.get_feature_names_out())
    assert "Preparation_Time_min" not in names and "Vehicle_Type_Bike" not in names
    assert len(names) == 1 + 5 + 3


def test_real_dataset_if_available():
    if not REAL_CSV.exists():
        return
    df = dl.load_csv(REAL_CSV)
    dl.validate_columns(df)
    data = dl.build_experiment_data(df)
    assert data.n_removed_missing_target == 0
    assert set(data.y_class.unique()) == set(dl.CLASS_ORDER)
    assert prep.missing_before_after(data.X_train, data.X_test, dl.FEATURE_COLUMNS, True)["after_train"] == 0
