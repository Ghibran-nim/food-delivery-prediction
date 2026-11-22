"""Aplikasi Streamlit - Analisis dan Prediksi Waktu Pengiriman Makanan.

File ini hanya mengatur tampilan dan navigasi. Seluruh logika penelitian
(preprocessing, model, ablation, prediksi) berada di folder `src/`.

Jalankan dengan:  streamlit run app.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import ablation as abl
from src import classification as clf_mod
from src import data_loader as dl
from src import eda
from src import prediction as pred
from src import preprocessing as prep
from src import regression as reg_mod

RESEARCH_TITLE = (
    "Analisis dan Prediksi Waktu Pengiriman Makanan Berdasarkan Karakteristik Pesanan "
    "dan Kondisi Pengantaran Menggunakan Machine Learning"
)
DEFAULT_CSV = Path(__file__).parent / "data" / "Food_Delivery_Times.csv"

CUSTOM_CSS = """
<style>
div[data-testid="stMetric"] {
    background-color: rgba(31, 78, 121, 0.08);
    border: 1px solid rgba(31, 78, 121, 0.25);
    padding: 12px 16px;
    border-radius: 10px;
}
</style>
"""


# ---------------------------------------------------------------------------
# Fungsi bantu
# ---------------------------------------------------------------------------
def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


def require_dataset():
    data = st.session_state.get("data")
    if data is None:
        st.warning("Dataset belum dimuat. Silakan buka halaman **Upload Dataset** terlebih dahulu.")
    return data


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def load_into_session(df: pd.DataFrame, source_name: str) -> bool:
    """Validasi dataset, bentuk split, dan latih model baseline."""
    try:
        data = dl.build_experiment_data(df)
    except dl.DatasetError as exc:
        st.error(str(exc))
        return False
    with st.spinner("Melatih model baseline (Decision Tree dan KNN)..."):
        models = pred.train_baseline_models(data)
    st.session_state["raw_df"] = df
    st.session_state["data"] = data
    st.session_state["models"] = models
    st.session_state["ablation"] = None
    st.session_state["source_name"] = source_name
    return True


# ---------------------------------------------------------------------------
# Halaman
# ---------------------------------------------------------------------------
def page_dashboard():
    st.title("🍔 Prediksi Waktu Pengiriman Makanan")
    st.caption(RESEARCH_TITLE)
    st.markdown(
        "Aplikasi ini mengikuti laporan Tugas Kecerdasan Artifisial. Dua model dengan tujuan berbeda "
        "dijalankan pada dua jalur terpisah:\n\n"
        "| Jalur | Algoritma | Keluaran | Standardisasi |\n|---|---|---|---|\n"
        "| 1 | Decision Tree Classifier | Kategori Fast / Medium / Slow | Tidak |\n"
        "| 2 | K-Nearest Neighbors Regressor | Estimasi waktu pengiriman (menit) | Ya (fitur numerik) |"
    )
    data = st.session_state.get("data")
    models = st.session_state.get("models")
    if data is None:
        st.info("Mulai dari halaman **Upload Dataset** untuk memuat `Food_Delivery_Times.csv`.")
        return

    st.subheader("Ringkasan dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah baris", f"{len(data.df):,}")
    c2.metric("Fitur prediktor", len(dl.FEATURE_COLUMNS))
    c3.metric("Data training", len(data.split.train_idx))
    c4.metric("Data testing", len(data.split.test_idx))

    st.subheader("Kinerja baseline pada data testing")
    st.markdown("**Decision Tree Classifier**")
    cols = st.columns(4)
    for col, (name, value) in zip(cols, models.clf.summary.items()):
        col.metric(name, f"{value:.4f}")
    st.markdown("**KNN Regressor**")
    cols = st.columns(3)
    for col, (name, value) in zip(cols, models.reg.summary.items()):
        col.metric(name, f"{value:.4f}")
    st.caption(
        "Nilai di atas dihitung langsung dari model yang dilatih pada dataset yang dimuat. "
        "Hasil hanya menggambarkan pola pada dataset ini dan bukan hubungan sebab-akibat."
    )

    with st.expander("Keterbatasan penelitian (sesuai laporan)"):
        st.markdown(
            "- Hanya menggunakan variabel yang tersedia dalam dataset.\n"
            "- Kondisi jalan, rute kurir, antrean restoran, dan kondisi lalu lintas real-time tidak tercakup.\n"
            "- Hasil tidak dapat langsung dianggap mewakili seluruh layanan pesan antar makanan di dunia nyata."
        )


def page_upload():
    st.header("Upload Dataset")
    st.write("Unggah file CSV *Food Delivery Time Prediction* dari Kaggle (`Food_Delivery_Times.csv`).")
    uploaded = st.file_uploader("Pilih file CSV", type=["csv"])

    if uploaded is not None:
        key = f"{uploaded.name}:{uploaded.size}"
        if st.session_state.get("upload_key") != key:
            try:
                df = dl.load_csv(uploaded)
            except dl.DatasetError as exc:
                st.error(str(exc))
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
        return
    st.success(f"Dataset aktif: **{st.session_state.get('source_name')}**")

    overview = dl.dataset_overview(raw)
    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah baris", f"{overview['n_rows']:,}")
    c2.metric("Jumlah kolom", overview["n_cols"])
    c3.metric("Baris duplikat", overview["n_duplicate_rows"])

    st.subheader("Kecocokan kolom dengan laporan")
    st.dataframe(dl.column_check(raw))
    st.subheader("Tipe data")
    st.dataframe(overview["dtypes"])
    st.subheader("Contoh data")
    st.dataframe(raw.head(10))


def page_eda():
    st.header("Exploratory Data Analysis")
    data = require_dataset()
    if data is None:
        return
    df = data.df
    st.caption("EDA bersifat deskriptif. Korelasi dan grafik tidak menunjukkan hubungan sebab-akibat.")
    tab_stat, tab_dist, tab_rel, tab_corr = st.tabs(
        ["Statistik deskriptif", "Distribusi", "Hubungan dengan waktu pengiriman", "Korelasi"]
    )

    with tab_stat:
        st.subheader("Statistik deskriptif (kolom numerik)")
        st.dataframe(eda.descriptive_statistics(df))
        st.subheader("Ringkasan kolom kategorikal")
        st.dataframe(eda.categorical_summary(df))

    with tab_dist:
        a, b = st.columns(2)
        with a:
            show_fig(eda.plot_histogram(df, dl.TARGET_COL, "Distribusi waktu pengiriman", "Waktu pengiriman (menit)"))
            show_fig(eda.plot_histogram(df, "Preparation_Time_min", "Distribusi waktu persiapan makanan",
                                        "Waktu persiapan (menit)", bins=25))
            show_fig(eda.plot_countplot(df, "Traffic_Level", "Distribusi tingkat kemacetan"))
            show_fig(eda.plot_countplot(df, "Vehicle_Type", "Distribusi jenis kendaraan"))
        with b:
            show_fig(eda.plot_histogram(df, "Distance_km", "Distribusi jarak pengiriman", "Jarak (km)"))
            show_fig(eda.plot_countplot(df, "Weather", "Distribusi cuaca"))
            show_fig(eda.plot_countplot(df, "Time_of_Day", "Distribusi waktu pemesanan"))
        st.caption("Nilai kosong ditampilkan sebagai '(kosong)' pada grafik kategorikal.")

    with tab_rel:
        a, b = st.columns(2)
        with a:
            show_fig(eda.plot_distance_vs_delivery(df))
            show_fig(eda.plot_box_by_category(df, "Weather", "Waktu pengiriman berdasarkan cuaca"))
        with b:
            show_fig(eda.plot_box_by_category(df, "Traffic_Level", "Waktu pengiriman berdasarkan tingkat kemacetan"))
        st.caption("Baris dengan nilai kosong pada kolom pengelompokan tidak diikutkan pada boxplot.")

    with tab_corr:
        st.dataframe(eda.numeric_correlation(df))
        show_fig(eda.plot_correlation_heatmap(df))
        st.caption("Korelasi hanya mengukur keterkaitan linear antarvariabel numerik, bukan sebab-akibat.")


def page_preprocessing():
    st.header("Data Preprocessing")
    data = require_dataset()
    if data is None:
        return
    raw = st.session_state["raw_df"]

    # 1. Kualitas data
    st.subheader("1. Pemeriksaan kualitas data")
    overview = dl.dataset_overview(raw)
    c1, c2, c3 = st.columns(3)
    c1.metric("Duplikat (seluruh kolom)", overview["n_duplicate_rows"])
    c2.metric("Duplikat (tanpa Order_ID)", overview["n_duplicate_ignoring_id"])
    c3.metric("Order_ID duplikat", overview["n_duplicate_order_id"])
    st.markdown("**Missing values per kolom (data asli)**")
    st.dataframe(dl.missing_report(raw))

    if data.n_removed_missing_target == 0:
        st.success("Delivery_Time_min tidak memiliki missing values. **0 baris dikeluarkan.** Target tidak pernah diimputasi.")
    else:
        st.warning(f"**{data.n_removed_missing_target} baris dikeluarkan** dari pelatihan. {data.removal_reason}")
    order_report = dl.order_id_report(raw)
    st.info(f"Order_ID: {order_report['n_missing']} missing, {order_report['n_duplicated']} duplikat. {order_report['note']}")
    st.markdown("**Tipe data**")
    st.dataframe(overview["dtypes"])
    st.markdown("**Outlier (metode IQR, deskriptif)**")
    st.dataframe(dl.outlier_report(data.df, dl.NUMERIC_FEATURES + [dl.TARGET_COL]))
    st.caption("Tidak ada data yang dihapus atau diubah karena outlier. Nilai ekstrem dapat berupa pengiriman yang memang lama.")

    # 2. Target
    st.subheader("2. Pembentukan target klasifikasi")
    target = dl.target_report(data.df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Minimum (menit)", f"{target['min']:g}")
    c2.metric("Maksimum (menit)", f"{target['max']:g}")
    c3.metric("Ada nilai desimal?", "Ya" if target["has_decimal"] else "Tidak")
    st.markdown(
        "| Kategori | Waktu pengiriman |\n|---|---|\n| Fast (Cepat) | ≤ 30 menit |\n"
        "| Medium (Sedang) | > 30 hingga 45 menit |\n| Slow (Lambat) | > 45 menit |"
    )
    distribution = dl.class_distribution_all(data)
    st.dataframe(distribution)
    show_fig(eda.plot_class_distribution(distribution))
    st.info(dl.imbalance_note(dl.class_distribution(data.y_class)))

    st.markdown("**Pembagian data**")
    split = data.split
    st.write(
        f"Training: **{len(split.train_idx)}** baris (80%), testing: **{len(split.test_idx)}** baris (20%), "
        f"random_state = **{split.random_state}**, stratified = **{'Ya' if split.stratified else 'Tidak'}**, "
        f"sidik jari split = `{split.fingerprint()}`."
    )
    st.caption(split.note + " Split yang sama dipakai oleh kedua model dan seluruh skenario Ablation Study.")

    # 3. Strategi
    st.subheader("3. Strategi preprocessing")
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

    # 4. Sebelum dan sesudah
    st.subheader("4. Missing values sebelum dan sesudah preprocessing")
    rep_dt = prep.missing_before_after(data.X_train, data.X_test, dl.FEATURE_COLUMNS, scale=False)
    rep_knn = prep.missing_before_after(data.X_train, data.X_test, dl.FEATURE_COLUMNS, scale=True)
    st.dataframe(rep_dt["table"])
    summary = pd.DataFrame({
        "Jalur": ["Decision Tree (tanpa standardisasi)", "KNN (dengan standardisasi)"],
        "Missing training sebelum": [rep_dt["before_train"], rep_knn["before_train"]],
        "Missing testing sebelum": [rep_dt["before_test"], rep_knn["before_test"]],
        "Missing training sesudah": [rep_dt["after_train"], rep_knn["after_train"]],
        "Missing testing sesudah": [rep_dt["after_test"], rep_knn["after_test"]],
        "Jumlah kolom setelah encoding": [rep_dt["n_columns_after"], rep_knn["n_columns_after"]],
    })
    st.dataframe(summary)

    st.markdown("**Nilai imputasi yang dipelajari dari data training**")
    imputation = prep.get_imputation_statistics(rep_dt["preprocessor"], dl.FEATURE_COLUMNS)
    imputation["Nilai (dari training)"] = imputation["Nilai (dari training)"].astype(str)  # campuran angka/teks
    st.dataframe(imputation)
    scaler_stats = prep.get_scaler_statistics(rep_knn["preprocessor"], dl.FEATURE_COLUMNS)
    st.markdown("**Parameter StandardScaler (hanya jalur KNN, dari data training)**")
    st.dataframe(scaler_stats)

    tab_dt, tab_knn = st.tabs(["Contoh data hasil: Decision Tree", "Contoh data hasil: KNN"])
    with tab_dt:
        st.dataframe(prep.transformed_frame(rep_dt["preprocessor"], data.X_train).head(10))
    with tab_knn:
        st.dataframe(prep.transformed_frame(rep_knn["preprocessor"], data.X_train).head(10))


def page_classification():
    st.header("Model Classification - Decision Tree Classifier")
    data = require_dataset()
    if data is None:
        return
    result = st.session_state["models"].clf
    st.write(
        "Memprediksi kategori waktu pengiriman (Fast / Medium / Slow) memakai 7 fitur prediktor, "
        "**tanpa standardisasi**. Parameter ditetapkan tetap."
    )
    with st.expander("Parameter model yang digunakan", expanded=True):
        st.dataframe(clf_mod.model_params_table(result.pipeline))

    st.subheader("Metrik pada data testing")
    cols = st.columns(4)
    for col, (name, value) in zip(cols, result.summary.items()):
        col.metric(name, f"{value:.4f}")
    majority, majority_acc = clf_mod.majority_baseline_accuracy(data.y_class_train, data.y_class_test)
    st.caption(
        f"Pembanding konteks: selalu menebak kelas terbanyak pada training ('{majority}') menghasilkan "
        f"Accuracy {majority_acc:.4f} pada data testing. Karena itu Accuracy perlu dibaca bersama metrik per kelas."
    )

    st.subheader("Precision, Recall, dan F1-score per kelas")
    st.dataframe(result.per_class.round(4))
    st.caption(
        "Macro average adalah rata-rata sederhana ketiga kelas, sehingga kelas kecil (Fast) berbobot sama "
        "dengan kelas besar (Slow). Ini dipilih karena distribusi kelas tidak seimbang."
    )

    a, b = st.columns(2)
    with a:
        st.subheader("Confusion Matrix")
        show_fig(clf_mod.plot_confusion_matrix(result.confusion))
    with b:
        st.subheader("Tabel Confusion Matrix")
        st.dataframe(result.confusion)
    st.subheader("Classification Report")
    st.code(result.report)
    with st.expander("Aturan keputusan pohon (teks)"):
        st.code(clf_mod.get_tree_rules(result.pipeline))


def page_regression():
    st.header("Model Regression - K-Nearest Neighbors Regressor")
    data = require_dataset()
    if data is None:
        return
    result = st.session_state["models"].reg
    st.write(
        "Memprediksi waktu pengiriman dalam menit memakai 7 fitur prediktor. Fitur numerik **distandardisasi** "
        "(parameter dari data training); hasil One-Hot Encoding tidak distandardisasi."
    )
    with st.expander("Parameter model yang digunakan", expanded=True):
        st.dataframe(reg_mod.model_params_table(result.pipeline))

    st.subheader("Metrik pada data testing")
    cols = st.columns(3)
    for col, (name, value) in zip(cols, result.summary.items()):
        col.metric(name, f"{value:.4f}")
    st.caption("MAE dan RMSE dalam menit (lebih rendah lebih baik). R² lebih tinggi umumnya lebih baik.")

    show_fig(reg_mod.plot_actual_vs_predicted(result))
    st.subheader("Perbandingan nilai aktual dan prediksi")
    st.dataframe(result.comparison)
    st.download_button("Unduh tabel (CSV)", csv_bytes(result.comparison.reset_index()), "aktual_vs_prediksi.csv", "text/csv")


def page_ablation():
    st.header("Ablation Study")
    data = require_dataset()
    if data is None:
        return
    st.write(
        "Setiap skenario menghilangkan satu komponen dari sistem lengkap (baseline). Split training-testing, "
        "random state, dan parameter model **sama** di semua skenario; imputasi, encoding, dan standardisasi "
        "**di-fit ulang pada data training** sesuai fitur yang tersisa. Total **15 pengujian** "
        "(Decision Tree 7 skenario, KNN 8 skenario)."
    )
    st.dataframe(abl.scenario_table())

    if st.button("Jalankan Ablation Study (15 pengujian)"):
        bar = st.progress(0.0)

        def update(done, total, text):
            bar.progress(done / total, text=f"{done}/{total}: {text}")

        st.session_state["ablation"] = abl.run_ablation(data, progress=update)

    result = st.session_state.get("ablation")
    if result is None:
        st.info("Tekan tombol di atas untuk menjalankan seluruh pengujian. Angka hanya muncul dari hasil eksekusi model.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Total pengujian model", result.n_runs)
    c2.metric("Data testing yang sama", result.n_test)
    c3.metric("Sidik jari split", result.split_fingerprint)

    tab_clf, tab_reg, tab_detail = st.tabs(["Decision Tree Classifier", "KNN Regressor", "Detail eksekusi"])
    with tab_clf:
        st.caption("Accuracy, Precision, Recall, F1-score: lebih tinggi umumnya lebih baik. Δ = skenario - baseline.")
        st.dataframe(result.clf_table.round(4))
        show_fig(abl.plot_classification_comparison(result.clf_table))
        st.subheader("Confusion Matrix per skenario")
        st.caption(
            "Dihitung dari prediksi model tiap skenario pada data testing yang sama. Baris = kelas aktual, "
            "kolom = kelas prediksi, urutan kelas: Fast, Medium, Slow. Eksperimen 7 tidak ada karena hanya menguji KNN."
        )
        show_fig(abl.plot_confusion_matrices(result.confusions, result.clf_table))
        with st.expander("Tabel angka Confusion Matrix per skenario"):
            for scenario_name, cm in result.confusions.items():
                st.markdown(f"**{scenario_name}**")
                st.dataframe(cm)
        st.download_button("Unduh Confusion Matrix semua skenario (CSV)",
                           csv_bytes(abl.confusion_long_table(result.confusions)),
                           "ablation_confusion_matrix.csv", "text/csv")
        st.subheader("Interpretasi")
        for line in abl.interpret_classification(result.clf_table, result.n_test):
            st.markdown(f"- {line}")
        st.download_button("Unduh tabel (CSV)", csv_bytes(result.clf_table), "ablation_classification.csv", "text/csv")
    with tab_reg:
        st.caption("MAE dan RMSE: lebih rendah umumnya lebih baik. R²: lebih tinggi umumnya lebih baik. Δ = skenario - baseline.")
        st.dataframe(result.reg_table.round(4))
        show_fig(abl.plot_regression_comparison(result.reg_table))
        st.subheader("Interpretasi")
        for line in abl.interpret_regression(result.reg_table):
            st.markdown(f"- {line}")
        st.download_button("Unduh tabel (CSV)", csv_bytes(result.reg_table), "ablation_regression.csv", "text/csv")
    with tab_detail:
        st.write("Bukti bahwa setiap pengujian memakai split dan parameter yang sama, dengan preprocessing di-fit ulang.")
        st.dataframe(result.details)


def page_prediction():
    st.header("Prediction")
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
        submitted = st.form_submit_button("Prediksi")

    if not submitted:
        return
    order = {
        "Distance_km": float(distance), "Weather": weather, "Traffic_Level": traffic,
        "Time_of_Day": time_of_day, "Vehicle_Type": vehicle,
        "Preparation_Time_min": float(prep_time), "Courier_Experience_yrs": float(experience),
    }
    result = pred.predict_new_order(models, order)
    a, b = st.columns(2)
    a.metric("Kategori waktu pengiriman (Decision Tree)", result["kategori"])
    b.metric("Estimasi waktu pengiriman (KNN)", f"{result['menit']:.1f} menit")
    for warning in pred.range_warnings(order, ranges):
        st.warning(warning)
    st.info(pred.DISCLAIMER)


# ---------------------------------------------------------------------------
# Navigasi
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


def main():
    st.set_page_config(page_title="Prediksi Waktu Pengiriman Makanan", page_icon="🍔", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.sidebar.title("Navigasi")
    page = st.sidebar.radio("Pilih bagian", list(PAGES.keys()))
    if st.session_state.get("data") is None:
        st.sidebar.warning("Dataset belum dimuat")
    else:
        st.sidebar.success(f"Dataset: {st.session_state.get('source_name')}")
    PAGES[page]()


if __name__ == "__main__":
    main()
