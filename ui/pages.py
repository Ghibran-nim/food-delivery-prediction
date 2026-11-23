"""Halaman aplikasi (lapisan UI).

Semua perhitungan dilakukan oleh fungsi di `src/` yang dipanggil persis seperti sebelumnya.
Modul ini hanya menata tampilan: layout, kartu, tabel, dan grafik bertema.
"""
from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import streamlit as st

from src import ablation as abl
from src import classification as clf_mod
from src import data_loader as dl
from src import eda
from src import prediction as pred
from src import preprocessing as prep
from src import regression as reg_mod
from ui import charts, theme
from ui import components as c

RESEARCH_TITLE = (
    "Analisis dan Prediksi Waktu Pengiriman Makanan Berdasarkan Karakteristik Pesanan "
    "dan Kondisi Pengantaran Menggunakan Machine Learning"
)
DEFAULT_CSV = Path(__file__).resolve().parent.parent / "data" / "Food_Delivery_Times.csv"

CLASS_BADGE = {"Fast": "success", "Medium": "warning", "Slow": "danger"}
DELTA_CLF = {f"Δ {m}": charts.HIGHER_IS_BETTER[m] for m in charts.CLF_METRICS}
DELTA_REG = {f"Δ {m}": charts.HIGHER_IS_BETTER[m] for m in charts.REG_METRICS}


# ---------------------------------------------------------------------------
# Fungsi bantu (logika dipertahankan dari versi sebelumnya)
# ---------------------------------------------------------------------------
def mode() -> str:
    return theme.resolve_theme(st.session_state.get("theme", "light"))


def require_dataset():
    data = st.session_state.get("data")
    if data is None:
        c.callout("warning", "Dataset belum dimuat. Silakan buka halaman Upload Dataset terlebih dahulu.")
    return data


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def load_into_session(df: pd.DataFrame, source_name: str) -> bool:
    """Validasi dataset, bentuk split, dan latih model baseline."""
    try:
        data = dl.build_experiment_data(df)
    except dl.DatasetError as exc:
        c.callout("danger", str(exc), title="Dataset tidak dapat digunakan")
        return False
    with st.spinner("Melatih model baseline (Decision Tree dan KNN)..."):
        models = pred.train_baseline_models(data)
    st.session_state["raw_df"] = df
    st.session_state["data"] = data
    st.session_state["models"] = models
    st.session_state["ablation"] = None
    st.session_state["source_name"] = source_name
    return True


def goto(page: str) -> None:
    """Callback tombol navigasi: pindah halaman lewat state radio sidebar."""
    st.session_state["nav_page"] = page


def _fmt(value, decimals: int = 4) -> str:
    return f"{value:.{decimals}f}"


def _metric_items(summary: dict, accent: bool = True) -> list[dict]:
    return [{"label": name, "value": _fmt(value), "accent": accent} for name, value in summary.items()]


# ---------------------------------------------------------------------------
# Halaman
# ---------------------------------------------------------------------------
def page_dashboard():
    c.page_header("Dashboard", RESEARCH_TITLE)
    c.section("Project Overview", "Dua model dengan tujuan berbeda dijalankan pada dua jalur terpisah (sesuai laporan).")
    c.kv_panel("Jalur 1 - Decision Tree Classifier", [
        ("Keluaran", "Kategori Fast / Medium / Slow"), ("Standardisasi", "Tidak")])
    c.kv_panel("Jalur 2 - K-Nearest Neighbors Regressor", [
        ("Keluaran", "Estimasi waktu pengiriman (menit)"), ("Standardisasi", "Ya (hanya fitur numerik)")])

    data = st.session_state.get("data")
    models = st.session_state.get("models")
    if data is None:
        c.callout("info", "Mulai dari halaman Upload Dataset untuk memuat Food_Delivery_Times.csv.")
        st.button("Buka Upload Dataset", type="primary", key="dash_upload", on_click=goto, args=("Upload Dataset",))
        return

    raw = st.session_state["raw_df"]
    missing_cells = int(raw.isna().sum().sum())
    rows_with_missing = int(raw.isna().any(axis=1).sum())
    c.section("Dataset Overview", f"Sumber: {st.session_state.get('source_name')}")
    c.metric_cards([
        {"label": "Total records", "value": f"{len(data.df):,}", "accent": True},
        {"label": "Fitur prediktor", "value": str(len(dl.FEATURE_COLUMNS))},
        {"label": "Data training", "value": str(len(data.split.train_idx)), "hint": "80%"},
        {"label": "Data testing", "value": str(len(data.split.test_idx)), "hint": "20%"},
        {"label": "Missing values", "value": str(missing_cells), "hint": f"{rows_with_missing} baris terdampak"},
    ])

    c.section("Classification Model Overview", "Decision Tree Classifier - kinerja baseline pada data testing")
    c.metric_cards(_metric_items(models.clf.summary))
    c.kv_panel("Parameter tetap", [(k, v) for k, v in clf_mod.DT_PARAMS.items()])

    c.section("Regression Model Overview", "KNN Regressor - kinerja baseline pada data testing")
    c.metric_cards(_metric_items(models.reg.summary))
    c.kv_panel("Parameter tetap", [(k, v) for k, v in reg_mod.KNN_PARAMS.items()])
    st.caption("Nilai dihitung langsung dari model yang dilatih pada dataset aktif. Hasil bukan hubungan sebab-akibat.")

    c.section("Navigasi cepat")
    targets = [n for n in PAGES if n != "Dashboard"]
    for start in range(0, len(targets), 4):
        cols = st.columns(4)
        for col, name in zip(cols, targets[start:start + 4]):
            with col:
                st.button(f"{ICONS[name]} {name}", key=f"goto_{name}", on_click=goto, args=(name,))

    with st.expander("Keterbatasan penelitian (sesuai laporan)"):
        st.markdown(
            "- Hanya menggunakan variabel yang tersedia dalam dataset.\n"
            "- Kondisi jalan, rute kurir, antrean restoran, dan kondisi lalu lintas real-time tidak tercakup.\n"
            "- Hasil tidak dapat langsung dianggap mewakili seluruh layanan pesan antar makanan di dunia nyata."
        )


def page_upload():
    c.page_header("Upload Dataset", "Unggah file CSV Food Delivery Time Prediction (Food_Delivery_Times.csv).")
    uploaded = st.file_uploader("Pilih file CSV", type=["csv"])

    if uploaded is not None:
        key = f"{uploaded.name}:{uploaded.size}"
        if st.session_state.get("upload_key") != key:
            try:
                df = dl.load_csv(uploaded)
            except dl.DatasetError as exc:
                c.callout("danger", str(exc), title="File tidak dapat dibaca")
                df = None
            if df is not None and load_into_session(df, uploaded.name):
                st.session_state["upload_key"] = key

    if DEFAULT_CSV.exists():
        if st.button("Gunakan file data/Food_Delivery_Times.csv"):
            if load_into_session(dl.load_csv(DEFAULT_CSV), DEFAULT_CSV.name) and uploaded is not None:
                # tandai file unggahan sebagai 'sudah diproses' agar tidak menimpa pilihan ini
                st.session_state["upload_key"] = f"{uploaded.name}:{uploaded.size}"

    raw = st.session_state.get("raw_df")
    if raw is None:
        c.callout("info", "Belum ada dataset. Unggah file CSV untuk memulai.")
        return
    c.callout("success", f"Dataset aktif: <b>{html.escape(str(st.session_state.get('source_name')))}</b>",
              allow_html=True)

    overview = dl.dataset_overview(raw)
    c.metric_cards([
        {"label": "Jumlah baris", "value": f"{overview['n_rows']:,}", "accent": True},
        {"label": "Jumlah kolom", "value": str(overview["n_cols"])},
        {"label": "Baris duplikat", "value": str(overview["n_duplicate_rows"])},
    ])
    c.section("Kecocokan kolom dengan laporan")
    c.show_table(dl.column_check(raw))
    c.section("Tipe data")
    c.show_table(overview["dtypes"])
    c.section("Contoh data")
    c.show_table(raw.head(10))


def page_eda():
    c.page_header("Exploratory Data Analysis", "Karakteristik dataset secara deskriptif.")
    data = require_dataset()
    if data is None:
        return
    df = data.df
    m = mode()
    st.caption("EDA bersifat deskriptif. Korelasi dan grafik tidak menunjukkan hubungan sebab-akibat.")
    tab_data, tab_stat, tab_missing, tab_dist, tab_rel, tab_corr = st.tabs(
        ["Dataset", "Statistik", "Missing Values", "Distribusi", "Fitur vs Target", "Korelasi"]
    )

    with tab_data:
        overview = dl.dataset_overview(st.session_state["raw_df"])
        c.metric_cards([
            {"label": "Jumlah baris", "value": f"{overview['n_rows']:,}", "accent": True},
            {"label": "Jumlah kolom", "value": str(overview["n_cols"])},
        ])
        c.section("Dataset preview")
        c.show_table(st.session_state["raw_df"].head(10))
        c.section("Dataset information")
        c.show_table(overview["dtypes"])

    with tab_stat:
        c.section("Statistik deskriptif (kolom numerik)")
        c.show_table(eda.descriptive_statistics(df), show_index=True)
        c.section("Ringkasan kolom kategorikal")
        c.show_table(eda.categorical_summary(df), max_height=420)

    with tab_missing:
        c.section("Missing values per kolom", "Dihitung pada data yang digunakan untuk pelatihan.")
        c.show_table(dl.missing_report(df))

    with tab_dist:
        a, b = st.columns(2)
        with a:
            c.show_fig(charts.histogram(df, dl.TARGET_COL, "Distribusi waktu pengiriman", "Waktu pengiriman (menit)", mode=m))
            c.show_fig(charts.histogram(df, "Preparation_Time_min", "Distribusi waktu persiapan makanan",
                                        "Waktu persiapan (menit)", bins=25, mode=m))
            c.show_fig(charts.countplot(df, "Traffic_Level", "Distribusi tingkat kemacetan", mode=m))
            c.show_fig(charts.countplot(df, "Vehicle_Type", "Distribusi jenis kendaraan", mode=m))
        with b:
            c.show_fig(charts.histogram(df, "Distance_km", "Distribusi jarak pengiriman", "Jarak (km)", mode=m))
            c.show_fig(charts.countplot(df, "Weather", "Distribusi cuaca", mode=m))
            c.show_fig(charts.countplot(df, "Time_of_Day", "Distribusi waktu pemesanan", mode=m))
        st.caption("Nilai kosong ditampilkan sebagai '(kosong)' pada grafik kategorikal.")

    with tab_rel:
        a, b = st.columns(2)
        with a:
            c.show_fig(charts.distance_vs_delivery(df, mode=m))
            c.show_fig(charts.box_by_category(df, "Weather", "Waktu pengiriman berdasarkan cuaca", mode=m))
        with b:
            c.show_fig(charts.box_by_category(df, "Traffic_Level", "Waktu pengiriman berdasarkan tingkat kemacetan", mode=m))
        st.caption("Baris dengan nilai kosong pada kolom pengelompokan tidak diikutkan pada boxplot.")

    with tab_corr:
        corr = eda.numeric_correlation(df)
        c.show_table(corr, show_index=True)
        c.show_fig(charts.correlation_heatmap(corr, mode=m))
        st.caption("Korelasi hanya mengukur keterkaitan linear antarvariabel numerik, bukan sebab-akibat.")


def page_preprocessing():
    c.page_header("Data Preprocessing", "Kualitas data, pembentukan target, pembagian data, dan strategi preprocessing.")
    data = require_dataset()
    if data is None:
        return
    raw = st.session_state["raw_df"]
    m = mode()

    c.section("1. Pemeriksaan kualitas data")
    overview = dl.dataset_overview(raw)
    c.metric_cards([
        {"label": "Duplikat (seluruh kolom)", "value": str(overview["n_duplicate_rows"])},
        {"label": "Duplikat (tanpa Order_ID)", "value": str(overview["n_duplicate_ignoring_id"])},
        {"label": "Order_ID duplikat", "value": str(overview["n_duplicate_order_id"])},
    ])
    st.markdown("**Missing values per kolom (data asli)**")
    c.show_table(dl.missing_report(raw))

    if data.n_removed_missing_target == 0:
        c.callout("success", "Delivery_Time_min tidak memiliki missing values. 0 baris dikeluarkan. Target tidak pernah diimputasi.")
    else:
        c.callout("warning", f"{data.n_removed_missing_target} baris dikeluarkan dari pelatihan. {data.removal_reason}")
    order_report = dl.order_id_report(raw)
    c.callout("info", f"Order_ID: {order_report['n_missing']} missing, {order_report['n_duplicated']} duplikat. {order_report['note']}")
    st.markdown("**Tipe data**")
    c.show_table(overview["dtypes"])
    st.markdown("**Outlier (metode IQR, deskriptif)**")
    c.show_table(dl.outlier_report(data.df, dl.NUMERIC_FEATURES + [dl.TARGET_COL]))
    st.caption("Tidak ada data yang dihapus atau diubah karena outlier. Nilai ekstrem dapat berupa pengiriman yang memang lama.")

    c.section("2. Pembentukan target klasifikasi")
    target = dl.target_report(data.df)
    c.metric_cards([
        {"label": "Minimum (menit)", "value": f"{target['min']:g}"},
        {"label": "Maksimum (menit)", "value": f"{target['max']:g}"},
        {"label": "Ada nilai desimal?", "value": "Ya" if target["has_decimal"] else "Tidak"},
    ])
    st.markdown(
        "| Kategori | Waktu pengiriman |\n|---|---|\n| Fast (Cepat) | ≤ 30 menit |\n"
        "| Medium (Sedang) | > 30 hingga 45 menit |\n| Slow (Lambat) | > 45 menit |"
    )
    distribution = dl.class_distribution_all(data)
    c.show_table(distribution)
    c.show_fig(charts.class_distribution(distribution, mode=m))
    c.callout("info", dl.imbalance_note(dl.class_distribution(data.y_class)))

    st.markdown("**Pembagian data**")
    split = data.split
    c.metric_cards([
        {"label": "Training", "value": str(len(split.train_idx)), "hint": "80%"},
        {"label": "Testing", "value": str(len(split.test_idx)), "hint": "20%"},
        {"label": "random_state", "value": str(split.random_state)},
        {"label": "Stratified", "value": "Ya" if split.stratified else "Tidak"},
        {"label": "Sidik jari split", "value": split.fingerprint()},
    ])
    st.caption(split.note + " Split yang sama dipakai oleh kedua model dan seluruh skenario Ablation Study.")

    c.section("3. Strategi preprocessing")
    st.markdown(
        "| Langkah | Decision Tree Classifier | KNN Regressor |\n|---|---|---|\n"
        "| Imputasi numerik | Median (dari training) | Median (dari training) |\n"
        "| Imputasi kategorikal | Modus (dari training) | Modus (dari training) |\n"
        "| Encoding kategorikal | One-Hot Encoding | One-Hot Encoding |\n"
        "| Standardisasi | **Tidak** | **Ya**, hanya pada fitur numerik |\n"
        "| Penghapusan baris | Tidak ada | Tidak ada |"
    )
    st.caption(
        "Semua langkah berada di dalam Pipeline/ColumnTransformer dan hanya di-fit pada data training. "
        "Hasil One-Hot Encoding tidak distandardisasi."
    )

    c.section("4. Missing values sebelum dan sesudah preprocessing")
    rep_dt = prep.missing_before_after(data.X_train, data.X_test, dl.FEATURE_COLUMNS, scale=False)
    rep_knn = prep.missing_before_after(data.X_train, data.X_test, dl.FEATURE_COLUMNS, scale=True)
    c.show_table(rep_dt["table"])
    summary = pd.DataFrame({
        "Jalur": ["Decision Tree (tanpa standardisasi)", "KNN (dengan standardisasi)"],
        "Missing training sebelum": [rep_dt["before_train"], rep_knn["before_train"]],
        "Missing testing sebelum": [rep_dt["before_test"], rep_knn["before_test"]],
        "Missing training sesudah": [rep_dt["after_train"], rep_knn["after_train"]],
        "Missing testing sesudah": [rep_dt["after_test"], rep_knn["after_test"]],
        "Jumlah kolom setelah encoding": [rep_dt["n_columns_after"], rep_knn["n_columns_after"]],
    })
    c.show_table(summary)

    st.markdown("**Nilai imputasi yang dipelajari dari data training**")
    imputation = prep.get_imputation_statistics(rep_dt["preprocessor"], dl.FEATURE_COLUMNS)
    imputation["Nilai (dari training)"] = imputation["Nilai (dari training)"].astype(str)  # campuran angka/teks
    c.show_table(imputation)
    scaler_stats = prep.get_scaler_statistics(rep_knn["preprocessor"], dl.FEATURE_COLUMNS)
    st.markdown("**Parameter StandardScaler (hanya jalur KNN, dari data training)**")
    c.show_table(scaler_stats)

    tab_dt, tab_knn = st.tabs(["Contoh data hasil: Decision Tree", "Contoh data hasil: KNN"])
    with tab_dt:
        c.show_table(prep.transformed_frame(rep_dt["preprocessor"], data.X_train).head(10))
    with tab_knn:
        c.show_table(prep.transformed_frame(rep_knn["preprocessor"], data.X_train).head(10))


def page_classification():
    c.page_header("Model Classification", "Decision Tree Classifier - evaluasi pada data testing.")
    data = require_dataset()
    if data is None:
        return
    result = st.session_state["models"].clf
    m = mode()

    c.section("Informasi model")
    left, right = st.columns(2)
    with left:
        c.kv_panel("Konfigurasi", [
            ("Algoritma", "DecisionTreeClassifier"), ("Target", "Fast / Medium / Slow"),
            ("Fitur prediktor", str(len(dl.FEATURE_COLUMNS))), ("Standardisasi", "Tidak"),
            ("Encoding", "One-Hot Encoding"),
        ])
    with right:
        c.kv_panel("Parameter tetap", [(k, v) for k, v in clf_mod.DT_PARAMS.items()])
    with st.expander("Seluruh parameter model"):
        c.show_table(clf_mod.model_params_table(result.pipeline))

    c.section("Metrik evaluasi", "Precision, Recall, dan F1-score adalah macro average dari tiga kelas.")
    c.metric_cards(_metric_items(result.summary))
    majority, majority_acc = clf_mod.majority_baseline_accuracy(data.y_class_train, data.y_class_test)
    st.caption(
        f"Pembanding konteks: selalu menebak kelas terbanyak pada training ('{majority}') menghasilkan "
        f"Accuracy {majority_acc:.4f} pada data testing. Karena itu Accuracy perlu dibaca bersama metrik per kelas."
    )

    c.section("Precision, Recall, dan F1-score per kelas")
    c.show_table(result.per_class.round(4), decimals=4)
    st.caption(
        "Macro average adalah rata-rata sederhana ketiga kelas, sehingga kelas kecil (Fast) berbobot sama "
        "dengan kelas besar (Slow). Ini dipilih karena distribusi kelas tidak seimbang."
    )

    c.section("Confusion Matrix", "Baris = kelas aktual, kolom = kelas prediksi. Urutan kelas: Fast, Medium, Slow.")
    a, b = st.columns(2)
    with a:
        c.show_fig(charts.confusion_matrix_figure(result.confusion, mode=m))
    with b:
        c.show_table(result.confusion, show_index=True)

    c.section("Classification Report")
    st.code(result.report, language=None)

    with st.expander("Hasil prediksi pada data testing"):
        predictions = pd.DataFrame({
            "Kelas aktual": result.y_true.values, "Kelas prediksi": result.y_pred,
        }, index=result.y_true.index).rename_axis("Indeks data")
        c.show_table(predictions, show_index=True, max_height=420)
    with st.expander("Aturan keputusan pohon (teks)"):
        st.code(clf_mod.get_tree_rules(result.pipeline), language=None)


def page_regression():
    c.page_header("Model Regression", "K-Nearest Neighbors Regressor - evaluasi pada data testing.")
    data = require_dataset()
    if data is None:
        return
    result = st.session_state["models"].reg
    m = mode()

    c.section("Informasi model")
    left, right = st.columns(2)
    with left:
        c.kv_panel("Konfigurasi", [
            ("Algoritma", "KNeighborsRegressor"), ("Target", "Delivery_Time_min (menit)"),
            ("Fitur prediktor", str(len(dl.FEATURE_COLUMNS))),
            ("Standardisasi", "Ya, hanya fitur numerik"), ("Encoding", "One-Hot Encoding (tidak distandardisasi)"),
        ])
    with right:
        c.kv_panel("Parameter tetap", [(k, v) for k, v in reg_mod.KNN_PARAMS.items()])
    with st.expander("Seluruh parameter model"):
        c.show_table(reg_mod.model_params_table(result.pipeline))

    c.section("Metrik evaluasi")
    c.metric_cards(_metric_items(result.summary))
    st.caption("MAE dan RMSE dalam menit (lebih rendah lebih baik). R² lebih tinggi umumnya lebih baik.")

    c.section("Aktual vs prediksi")
    c.show_fig(charts.actual_vs_predicted(result.y_true, result.y_pred, mode=m))

    c.section("Perbandingan nilai aktual dan prediksi")
    c.show_table(result.comparison.rename_axis("Indeks data"), show_index=True, max_height=460)
    st.download_button("Unduh tabel (CSV)", csv_bytes(result.comparison.reset_index()), "aktual_vs_prediksi.csv", "text/csv")


def page_ablation():
    c.page_header("Ablation Study", "Perbandingan 15 pengujian terhadap baseline pada data testing yang sama.")
    data = require_dataset()
    if data is None:
        return
    m = mode()

    # 1. Experiment Overview
    c.section("1. Experiment Overview",
              "Split training-testing, random state, dan parameter model sama di semua skenario; imputasi, encoding, "
              "dan standardisasi di-fit ulang pada data training sesuai fitur yang tersisa.")
    scenarios = abl.define_scenarios()
    n_clf = sum(abl.CLF_NAME in s.models for s in scenarios)
    n_reg = sum(abl.REG_NAME in s.models for s in scenarios)
    c.metric_cards([
        {"label": "Total pengujian", "value": str(n_clf + n_reg), "accent": True},
        {"label": "Skenario Decision Tree", "value": str(n_clf)},
        {"label": "Skenario KNN Regressor", "value": str(n_reg), "hint": "termasuk Eksperimen 7"},
    ])
    c.show_table(abl.scenario_table())

    if st.button("Jalankan Ablation Study (15 pengujian)", type="primary"):
        bar = st.progress(0.0)

        def update(done, total, text):
            bar.progress(done / total, text=f"{done}/{total}: {text}")

        st.session_state["ablation"] = abl.run_ablation(data, progress=update)

    result = st.session_state.get("ablation")
    if result is None:
        c.callout("info", "Tekan tombol di atas untuk menjalankan seluruh pengujian. Angka hanya muncul dari hasil eksekusi model.")
        return

    c.metric_cards([
        {"label": "Total pengujian model", "value": str(result.n_runs), "accent": True},
        {"label": "Data testing yang sama", "value": str(result.n_test)},
        {"label": "Sidik jari split", "value": result.split_fingerprint},
    ])

    # 2. Decision Tree Results
    c.section("2. Decision Tree Results", "Accuracy, Precision, Recall, F1-score: lebih tinggi umumnya lebih baik.")
    clf_cols = ["Skenario", "Komponen yang dihilangkan", "Jumlah fitur"] + charts.CLF_METRICS
    c.show_table(result.clf_table[clf_cols].round(4), decimals=4)

    # 3. KNN Regression Results
    c.section("3. KNN Regression Results", "MAE dan RMSE: lebih rendah umumnya lebih baik. R²: lebih tinggi umumnya lebih baik.")
    reg_cols = ["Skenario", "Komponen yang dihilangkan", "Jumlah fitur"] + charts.REG_METRICS
    c.show_table(result.reg_table[reg_cols].round(4), decimals=4)

    # 4. Baseline Comparison
    c.section("4. Baseline Comparison",
              "Selisih = skenario - baseline. Panah menunjukkan arah perubahan; hijau = menuju performa lebih baik, "
              "merah = menuju performa lebih buruk.")
    tab_c, tab_r = st.tabs(["Decision Tree Classifier", "KNN Regressor"])
    with tab_c:
        delta_cols = ["Skenario", "Komponen yang dihilangkan"] + list(DELTA_CLF)
        c.show_table(result.clf_table[delta_cols].round(4), delta_cols=DELTA_CLF)
        st.markdown("**Interpretasi**")
        for line in abl.interpret_classification(result.clf_table, result.n_test):
            st.markdown(f"- {line}")
    with tab_r:
        delta_cols = ["Skenario", "Komponen yang dihilangkan"] + list(DELTA_REG)
        c.show_table(result.reg_table[delta_cols].round(4), delta_cols=DELTA_REG)
        st.markdown("**Interpretasi**")
        for line in abl.interpret_regression(result.reg_table):
            st.markdown(f"- {line}")

    # 5. Performance Visualization
    c.section("5. Performance Visualization", "Batang berwarna pekat = baseline; garis putus-putus = nilai baseline.")
    tab_c, tab_r = st.tabs(["Decision Tree Classifier", "KNN Regressor"])
    with tab_c:
        c.show_fig(charts.ablation_classification(result.clf_table, mode=m))
    with tab_r:
        c.show_fig(charts.ablation_regression(result.reg_table, mode=m))

    # 6. Decision Tree Confusion Matrix Grid
    c.section("6. Decision Tree Confusion Matrix Grid",
              "Dihitung dari prediksi model tiap skenario pada data testing yang sama. Baris = kelas aktual, "
              "kolom = kelas prediksi, urutan kelas: Fast, Medium, Slow. Eksperimen 7 tidak ada karena hanya menguji KNN.")
    c.show_fig(charts.confusion_grid(result.confusions, result.clf_table, mode=m))
    with st.expander("Tabel angka Confusion Matrix per skenario"):
        for scenario_name, cm in result.confusions.items():
            st.markdown(f"**{scenario_name}**")
            c.show_table(cm, show_index=True)

    # 7. Experiment Details dan Download Results
    c.section("7. Experiment Details dan Download Results")
    with st.expander("Detail eksekusi 15 pengujian"):
        st.write("Bukti bahwa setiap pengujian memakai split dan parameter yang sama, dengan preprocessing di-fit ulang.")
        c.show_table(result.details)
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button("Unduh tabel Decision Tree (CSV)", csv_bytes(result.clf_table), "ablation_classification.csv", "text/csv")
    with d2:
        st.download_button("Unduh tabel KNN (CSV)", csv_bytes(result.reg_table), "ablation_regression.csv", "text/csv")
    with d3:
        st.download_button("Unduh Confusion Matrix (CSV)", csv_bytes(abl.confusion_long_table(result.confusions)),
                           "ablation_confusion_matrix.csv", "text/csv")


def page_prediction():
    c.page_header("Prediction", "Estimasi kategori dan waktu pengiriman untuk pesanan baru.")
    data = require_dataset()
    if data is None:
        return
    models = st.session_state["models"]
    options = pred.get_form_options(data.X_train)
    ranges = options["ranges"]
    st.write("Masukkan karakteristik pesanan baru. Prediksi memakai model dan pipeline baseline yang sudah dilatih.")

    with st.form("form_prediksi"):
        c1, c2 = st.columns(2)
        with c1:
            distance = st.number_input("Distance_km (km)", min_value=0.0, value=round(sum(ranges["Distance_km"]) / 2, 1), step=0.5)
            prep_time = st.number_input("Preparation_Time_min (menit)", min_value=0.0,
                                        value=float(round(sum(ranges["Preparation_Time_min"]) / 2)), step=1.0)
            experience = st.number_input("Courier_Experience_yrs (tahun)", min_value=0.0,
                                         value=float(round(sum(ranges["Courier_Experience_yrs"]) / 2)), step=1.0)
        with c2:
            weather = st.selectbox("Weather", options["categories"]["Weather"])
            traffic = st.selectbox("Traffic_Level", options["categories"]["Traffic_Level"])
            time_of_day = st.selectbox("Time_of_Day", options["categories"]["Time_of_Day"])
            vehicle = st.selectbox("Vehicle_Type", options["categories"]["Vehicle_Type"])
        submitted = st.form_submit_button("Prediksi", type="primary")

    if not submitted:
        return
    order = {
        "Distance_km": float(distance), "Weather": weather, "Traffic_Level": traffic,
        "Time_of_Day": time_of_day, "Vehicle_Type": vehicle,
        "Preparation_Time_min": float(prep_time), "Courier_Experience_yrs": float(experience),
    }
    result = pred.predict_new_order(models, order)
    c.section("Hasil prediksi")
    a, b = st.columns(2)
    with a:
        c.result_card("Kategori waktu pengiriman (Decision Tree)", result["kategori"],
                      badge_html=c.badge(result["kategori"], CLASS_BADGE.get(result["kategori"], "neutral")))
    with b:
        c.result_card("Estimasi waktu pengiriman (KNN)", f"{result['menit']:.1f} menit")
    for warning in pred.range_warnings(order, ranges):
        c.callout("warning", warning)
    c.callout("info", pred.DISCLAIMER)


# ---------------------------------------------------------------------------
# Navigasi dan tema
# ---------------------------------------------------------------------------
PAGES = {
    "Dashboard": page_dashboard,
    "Upload Dataset": page_upload,
    "Exploratory Data Analysis": page_eda,
    "Data Preprocessing": page_preprocessing,
    "Model Classification": page_classification,
    "Model Regression": page_regression,
    "Ablation Study": page_ablation,
    "Prediction": page_prediction,
}
ICONS = {
    "Dashboard": ":material/space_dashboard:",
    "Upload Dataset": ":material/upload_file:",
    "Exploratory Data Analysis": ":material/bar_chart:",
    "Data Preprocessing": ":material/tune:",
    "Model Classification": ":material/account_tree:",
    "Model Regression": ":material/show_chart:",
    "Ablation Study": ":material/science:",
    "Prediction": ":material/online_prediction:",
}


def init_state() -> str:
    """Menentukan tema awal (query param ?theme=dark|light, default light) dan halaman awal."""
    if "theme" not in st.session_state:
        st.session_state["theme"] = theme.resolve_theme(st.query_params.get("theme", "light"))
        st.session_state["theme_dark"] = st.session_state["theme"] == "dark"
    st.session_state.setdefault("nav_page", "Dashboard")
    return st.session_state["theme"]


def _on_theme_toggle() -> None:
    new_mode = "dark" if st.session_state.get("theme_dark") else "light"
    st.session_state["theme"] = new_mode
    st.query_params["theme"] = new_mode  # agar tema bertahan saat halaman di-refresh


def render_sidebar() -> str:
    """Sidebar: branding, navigasi, pemilih tema, dan status dataset. Mengembalikan halaman terpilih."""
    with st.sidebar:
        c.brand()
        c.side_label("Navigasi")
        st.radio("Navigasi", list(PAGES), format_func=lambda name: f"{ICONS[name]} {name}",
                 key="nav_page", label_visibility="collapsed")
        c.side_label("Tampilan")
        st.toggle("Mode gelap", key="theme_dark", on_change=_on_theme_toggle,
                  help="Beralih antara Light Mode dan Dark Mode.")
        c.side_label("Dataset")
        if st.session_state.get("data") is None:
            c.html(c.badge("Belum dimuat", "warning"))
        else:
            c.html(c.badge(str(st.session_state.get("source_name")), "success"))
    return st.session_state["nav_page"]
