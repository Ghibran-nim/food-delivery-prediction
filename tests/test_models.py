"""Unit test: Decision Tree, KNN, metrik evaluasi, dan prediksi pesanan baru."""
import numpy as np
import pandas as pd

from tests.helpers import REAL_CSV, make_synthetic_df  # noqa: F401  (juga mengatur sys.path)
from src import classification as clf
from src import data_loader as dl
from src import prediction as pred
from src import preprocessing as prep
from src import regression as reg


def _data():
    return dl.build_experiment_data(make_synthetic_df())


def _expect_raises(exc, func, *args):
    try:
        func(*args)
    except exc:
        return
    raise AssertionError(f"Seharusnya muncul {exc.__name__}")


# --- Konfigurasi tetap ------------------------------------------------------
def test_fixed_model_parameters():
    dt = clf.build_decision_tree()
    assert dt.max_depth == 5 and dt.random_state == 42 and dt.criterion == "gini"
    knn = reg.build_knn()
    assert knn.n_neighbors == 5 and knn.weights == "uniform" and knn.metric == "minkowski" and knn.p == 2
    assert type(dt).__name__ == "DecisionTreeClassifier" and type(knn).__name__ == "KNeighborsRegressor"


# --- Decision Tree (tanpa standardisasi) ------------------------------------
def test_decision_tree_runs_without_scaler_and_predicts_valid_labels():
    data = _data()
    pipe = clf.train_classifier(data.X_train, data.y_class_train)
    assert not prep.has_scaler(pipe) and prep.has_encoder(pipe)
    y_pred = pipe.predict(data.X_test[dl.FEATURE_COLUMNS])
    assert len(y_pred) == len(data.X_test) and set(y_pred) <= set(dl.CLASS_ORDER)


# --- KNN (dengan standardisasi) ---------------------------------------------
def test_knn_runs_with_scaler_and_predicts_numeric():
    data = _data()
    pipe = reg.train_regressor(data.X_train, data.y_reg_train)
    assert prep.has_scaler(pipe) and prep.has_encoder(pipe)
    y_pred = pipe.predict(data.X_test[dl.FEATURE_COLUMNS])
    assert len(y_pred) == len(data.X_test) and np.isfinite(y_pred).all()


def test_knn_without_scaler_option_keeps_encoder():
    data = _data()
    pipe = reg.train_regressor(data.X_train, data.y_reg_train, scale=False)
    assert not prep.has_scaler(pipe) and prep.has_encoder(pipe)


# --- Metrik dengan nilai yang dihitung manual -------------------------------
def test_classification_metrics_known_values():
    y_true = ["Fast", "Fast", "Medium", "Medium", "Slow", "Slow", "Slow", "Slow"]
    y_pred = ["Fast", "Medium", "Medium", "Medium", "Slow", "Slow", "Slow", "Fast"]
    m = clf.compute_classification_metrics(y_true, y_pred)
    per = m["per_class"].set_index("Kelas")
    # Fast: TP=1, FP=1 (Slow->Fast), FN=1  | Medium: TP=2, FP=1, FN=0 | Slow: TP=3, FP=0, FN=1
    assert np.isclose(per.loc["Fast", "Precision"], 1 / 2) and np.isclose(per.loc["Fast", "Recall"], 1 / 2)
    assert np.isclose(per.loc["Medium", "Precision"], 2 / 3) and np.isclose(per.loc["Medium", "Recall"], 1.0)
    assert np.isclose(per.loc["Slow", "Precision"], 1.0) and np.isclose(per.loc["Slow", "Recall"], 3 / 4)
    assert np.isclose(m["summary"]["Accuracy"], 6 / 8)
    assert np.isclose(m["summary"]["Precision (macro)"], (1 / 2 + 2 / 3 + 1) / 3)
    assert np.isclose(m["summary"]["Recall (macro)"], (1 / 2 + 1 + 3 / 4) / 3)
    f1 = [2 * 0.5 * 0.5 / 1.0, 2 * (2 / 3) * 1 / (2 / 3 + 1), 2 * 1 * 0.75 / 1.75]
    assert np.isclose(m["summary"]["F1-score (macro)"], np.mean(f1))


def test_regression_metrics_known_values():
    m = reg.compute_regression_metrics([10, 20, 30], [12, 18, 33])
    assert np.isclose(m["MAE"], 7 / 3)
    assert np.isclose(m["RMSE"], np.sqrt(17 / 3))
    assert np.isclose(m["R²"], 1 - 17 / 200)


def test_evaluation_results_are_complete():
    data = _data()
    models = pred.train_baseline_models(data)
    c, r = models.clf, models.reg
    assert set(c.summary) == {"Accuracy", "Precision (macro)", "Recall (macro)", "F1-score (macro)"}
    assert list(c.per_class["Kelas"]) == dl.CLASS_ORDER
    assert c.confusion.values.sum() == len(data.X_test)
    assert "precision" in c.report and "recall" in c.report
    assert set(r.summary) == {"MAE", "RMSE", "R²"}
    assert len(r.comparison) == len(data.X_test)
    assert {"Aktual (menit)", "Prediksi (menit)"} <= set(r.comparison.columns)


def test_training_is_deterministic():
    data = _data()
    a = pred.train_baseline_models(data)
    b = pred.train_baseline_models(data)
    assert a.clf.summary == b.clf.summary and a.reg.summary == b.reg.summary


def test_majority_baseline_accuracy():
    majority, acc = clf.majority_baseline_accuracy(["Slow", "Slow", "Fast"], ["Slow", "Fast", "Slow", "Slow"])
    assert majority == "Slow" and np.isclose(acc, 0.75)


# --- Prediksi data baru -----------------------------------------------------
ORDER = {
    "Distance_km": 8.5, "Weather": "Rainy", "Traffic_Level": "High", "Time_of_Day": "Evening",
    "Vehicle_Type": "Bike", "Preparation_Time_min": 15, "Courier_Experience_yrs": 3.0,
}


def test_predict_new_order_uses_trained_pipelines():
    data = _data()
    models = pred.train_baseline_models(data)
    out = pred.predict_new_order(models, ORDER)
    frame = pd.DataFrame([ORDER])[dl.FEATURE_COLUMNS]
    assert out["kategori"] in dl.CLASS_ORDER
    assert out["kategori"] == models.clf.pipeline.predict(frame)[0]
    assert np.isclose(out["menit"], models.reg.pipeline.predict(frame)[0])


def test_predict_new_order_validates_input():
    bad = dict(ORDER)
    bad.pop("Weather")
    _expect_raises(ValueError, pred.build_order_frame, bad)
    _expect_raises(ValueError, pred.build_order_frame, {**ORDER, "Order_ID": 1})


def test_form_options_and_range_warnings():
    data = _data()
    opts = pred.get_form_options(data.X_train)
    assert set(opts["categories"]) == set(dl.CATEGORICAL_FEATURES)
    assert "Bike" in opts["categories"]["Vehicle_Type"]
    assert pred.range_warnings(ORDER, opts["ranges"]) == []
    assert len(pred.range_warnings({**ORDER, "Distance_km": 500}, opts["ranges"])) == 1


def test_real_dataset_models_if_available():
    if not REAL_CSV.exists():
        return
    data = dl.build_experiment_data(dl.load_csv(REAL_CSV))
    models = pred.train_baseline_models(data)
    assert 0 <= models.clf.summary["Accuracy"] <= 1 and models.reg.summary["MAE"] > 0
