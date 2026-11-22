"""Unit test: Ablation Study (15 pengujian)."""
import numpy as np

from tests.helpers import REAL_CSV, make_synthetic_df  # noqa: F401  (juga mengatur sys.path)
from src import ablation as abl
from src import data_loader as dl
from src import preprocessing as prep
from src import prediction as pred

_CACHE = {}


def _run():
    """Ablation dijalankan sekali lalu dipakai ulang oleh semua test di file ini."""
    if "result" not in _CACHE:
        data = dl.build_experiment_data(make_synthetic_df(n=500, seed=1))
        _CACHE["data"] = data
        _CACHE["result"] = abl.run_ablation(data)
    return _CACHE["data"], _CACHE["result"]


def test_scenario_definitions_match_report():
    sc = {s.key: s for s in abl.define_scenarios()}
    assert list(sc) == ["baseline", "exp1", "exp2", "exp3", "exp4", "exp5", "exp6", "exp7"]
    removed = {
        "exp1": {"Distance_km"}, "exp2": {"Weather", "Traffic_Level"}, "exp3": {"Time_of_Day"},
        "exp4": {"Vehicle_Type"}, "exp5": {"Preparation_Time_min"}, "exp6": {"Courier_Experience_yrs"},
    }
    for key, cols in removed.items():
        assert set(dl.FEATURE_COLUMNS) - set(sc[key].features) == cols
        assert set(sc[key].models) == {abl.CLF_NAME, abl.REG_NAME}
    assert set(sc["baseline"].features) == set(dl.FEATURE_COLUMNS) and sc["baseline"].scale
    # Eksperimen 7: semua fitur tetap, hanya standardisasi dihilangkan, hanya KNN
    assert set(sc["exp7"].features) == set(dl.FEATURE_COLUMNS) and not sc["exp7"].scale
    assert sc["exp7"].models == (abl.REG_NAME,)


def test_total_of_15_runs_7_classification_8_regression():
    _, res = _run()
    assert len(res.clf_table) == 7 and len(res.reg_table) == 8 and res.n_runs == 15
    assert len(res.details) == 15 and len(res.pipelines) == 15
    assert "Eksperimen 7" not in set(res.clf_table["Skenario"])
    assert "Eksperimen 7" in set(res.reg_table["Skenario"])


def test_all_runs_use_same_split_and_same_parameters():
    data, res = _run()
    assert res.split_fingerprint == data.split.fingerprint()
    assert set(res.details["Sidik jari split"]) == {res.split_fingerprint}
    assert set(res.details["Jumlah data testing"]) == {len(data.split.test_idx)}
    for model_name in (abl.CLF_NAME, abl.REG_NAME):
        params = set(res.details.loc[res.details["Model"] == model_name, "Parameter model"])
        assert len(params) == 1, "Parameter model harus identik di semua skenario"
    assert "max_depth=5" in set(res.details.loc[res.details["Model"] == abl.CLF_NAME, "Parameter model"]).pop()
    assert "n_neighbors=5" in set(res.details.loc[res.details["Model"] == abl.REG_NAME, "Parameter model"]).pop()


def test_each_scenario_refits_preprocessing_on_remaining_features():
    data, res = _run()
    objects = set()
    for (key, model_name), pipe in res.pipelines.items():
        sc = next(s for s in abl.define_scenarios() if s.key == key)
        pre = pipe.named_steps["preprocessor"]
        objects.add(id(pre))
        numeric, categorical = prep.split_feature_types(sc.features)
        expected_cols = len(numeric) + sum(data.X_train[c].nunique() for c in categorical)
        assert len(pre.get_feature_names_out()) == expected_cols
        assert set(pipe.feature_names_in_) == set(sc.features)
        # statistik imputer berasal dari training pada fitur yang tersisa
        if numeric:
            imputer = pre.named_transformers_["num"].named_steps["imputer"]
            assert np.allclose(imputer.statistics_, data.X_train[numeric].median().values)
    assert len(objects) == 15, "Setiap pengujian harus memakai preprocessor baru"


def test_encoding_present_in_every_scenario_and_dt_never_scaled():
    _, res = _run()
    for (key, model_name), pipe in res.pipelines.items():
        assert prep.has_encoder(pipe), f"Encoding hilang pada {key} / {model_name}"
        if model_name == abl.CLF_NAME:
            assert not prep.has_scaler(pipe)
        elif key == "exp7":
            assert not prep.has_scaler(pipe)
        else:
            assert prep.has_scaler(pipe)


def test_experiment_7_differs_from_baseline_only_in_scaler():
    _, res = _run()
    base = res.pipelines[("baseline", abl.REG_NAME)]
    exp7 = res.pipelines[("exp7", abl.REG_NAME)]
    assert list(base.feature_names_in_) == list(exp7.feature_names_in_)
    assert prep.has_scaler(base) and not prep.has_scaler(exp7)
    assert base.named_steps["model"].get_params() == exp7.named_steps["model"].get_params()


def test_baseline_in_ablation_equals_standalone_baseline():
    data, res = _run()
    models = pred.train_baseline_models(data)
    for metric, value in models.clf.summary.items():
        assert np.isclose(res.clf_table.iloc[0][metric], value)
    for metric, value in models.reg.summary.items():
        assert np.isclose(res.reg_table.iloc[0][metric], value)


def test_deltas_are_scenario_minus_baseline():
    _, res = _run()
    for table, metrics in ((res.clf_table, abl.CLF_METRICS), (res.reg_table, abl.REG_METRICS)):
        assert table.iloc[0]["Skenario"] == "Baseline"
        for metric in metrics:
            expected = table[metric] - table.iloc[0][metric]
            assert np.allclose(table[f"Δ {metric}"], expected)
            assert table.iloc[0][f"Δ {metric}"] == 0


def test_results_come_from_model_execution_not_constants():
    """Pembuktian sederhana: mengubah data training mengubah hasil, jadi angka bukan hard-coded."""
    data_a = dl.build_experiment_data(make_synthetic_df(n=500, seed=1))
    data_b = dl.build_experiment_data(make_synthetic_df(n=500, seed=2))
    res_a, res_b = abl.run_ablation(data_a), abl.run_ablation(data_b)
    assert not np.allclose(res_a.reg_table["MAE"], res_b.reg_table["MAE"])


def test_interpretation_is_objective_and_not_causal():
    data, res = _run()
    n_test = len(data.split.test_idx)
    text = " ".join(abl.interpret_classification(res.clf_table, n_test) + abl.interpret_regression(res.reg_table)).lower()
    assert "bukan hubungan sebab-akibat" in text
    for forbidden in ("menyebabkan", "penyebab", "paling berpengaruh", "mengakibatkan"):
        assert forbidden not in text
    assert abl.describe_change("MAE", 5.0, 8.0).endswith("performa lebih buruk dari baseline)")
    assert abl.describe_change("R²", 0.5, 0.7).endswith("performa lebih baik dari baseline)")
    assert "tidak berubah" in abl.describe_change("Accuracy", 0.8, 0.8)


def test_figures_are_created():
    import matplotlib.pyplot as plt
    _, res = _run()
    for fig in (abl.plot_classification_comparison(res.clf_table), abl.plot_regression_comparison(res.reg_table)):
        assert fig is not None
        plt.close(fig)


def test_confusion_matrix_exists_only_for_decision_tree_scenarios():
    _, res = _run()
    expected = ["Baseline"] + [f"Eksperimen {i}" for i in range(1, 7)]
    assert list(res.confusions) == expected and len(res.confusions) == 7
    assert "Eksperimen 7" not in res.confusions
    assert list(res.confusions) == list(res.clf_table["Skenario"])


def test_confusion_matrix_label_order_and_shape():
    from src.data_loader import CLASS_ORDER
    _, res = _run()
    for name, cm in res.confusions.items():
        assert cm.shape == (3, 3), name
        assert list(cm.index) == [f"Aktual: {c}" for c in CLASS_ORDER], name
        assert list(cm.columns) == [f"Prediksi: {c}" for c in CLASS_ORDER], name


def test_confusion_matrix_matches_each_scenarios_own_predictions_on_test_set():
    """Dihitung ulang secara independen dari pipeline masing-masing skenario pada data testing."""
    from sklearn.metrics import confusion_matrix
    data, res = _run()
    for sc in abl.define_scenarios():
        if abl.CLF_NAME not in sc.models:
            continue
        features = list(sc.features)
        y_pred = res.pipelines[(sc.key, abl.CLF_NAME)].predict(data.X_test[features])
        expected = confusion_matrix(data.y_class_test, y_pred, labels=dl.CLASS_ORDER)
        assert (res.confusions[sc.name].values == expected).all(), sc.name


def test_confusion_matrix_uses_test_set_not_training_set():
    data, res = _run()
    n_test, n_train = len(data.split.test_idx), len(data.split.train_idx)
    test_support = data.y_class_test.value_counts().reindex(dl.CLASS_ORDER).values
    for name, cm in res.confusions.items():
        assert cm.values.sum() == n_test and cm.values.sum() != n_train, name
        assert (cm.values.sum(axis=1) == test_support).all(), name   # jumlah per baris = support kelas aktual di testing


def test_confusion_matrix_consistent_with_reported_metrics():
    _, res = _run()
    for _, row in res.clf_table.iterrows():
        cm = res.confusions[row["Skenario"]].values
        assert np.isclose(np.trace(cm) / cm.sum(), row["Accuracy"])
        assert np.isclose(np.mean(np.diag(cm) / cm.sum(axis=1)), row["Recall (macro)"])
        assert np.isclose(np.mean(np.diag(cm) / np.maximum(cm.sum(axis=0), 1)), row["Precision (macro)"])


def test_baseline_confusion_in_ablation_equals_standalone_baseline():
    data, res = _run()
    models = pred.train_baseline_models(data)
    assert (res.confusions["Baseline"].values == models.clf.confusion.values).all()
    assert list(res.confusions["Baseline"].index) == list(models.clf.confusion.index)


def test_confusion_long_table_and_grid_figure():
    import matplotlib.pyplot as plt
    _, res = _run()
    long = abl.confusion_long_table(res.confusions)
    assert len(long) == 7 * 9 and set(long["Skenario"]) == set(res.confusions)
    assert long.groupby("Skenario")["Jumlah"].sum().eq(res.n_test).all()
    fig = abl.plot_confusion_matrices(res.confusions, res.clf_table)
    assert len(fig.axes) == 8   # 7 panel + 1 panel kosong
    plt.close(fig)


def test_real_dataset_full_ablation_if_available():
    if not REAL_CSV.exists():
        return
    data = dl.build_experiment_data(dl.load_csv(REAL_CSV))
    res = abl.run_ablation(data)
    assert res.n_runs == 15 and len(res.clf_table) == 7 and len(res.reg_table) == 8
    assert len(set(res.details["Sidik jari split"])) == 1
