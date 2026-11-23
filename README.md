# Prediksi Waktu Pengiriman Makanan

Aplikasi machine learning berbasis Streamlit untuk **Tugas Kecerdasan Artifisial**:

> *Analisis dan Prediksi Waktu Pengiriman Makanan Berdasarkan Karakteristik Pesanan dan Kondisi Pengantaran Menggunakan Machine Learning*

## 1. Deskripsi proyek

Aplikasi membaca dataset pengiriman makanan, melakukan eksplorasi dan preprocessing, melatih dua model pada dua jalur terpisah, mengevaluasinya, menjalankan Ablation Study, lalu menyediakan formulir prediksi untuk pesanan baru. Seluruh metodologi mengikuti laporan tugas.

## 2. Tujuan penelitian

- Menganalisis karakteristik dataset pengiriman makanan.
- Mengidentifikasi pola yang berkaitan dengan waktu pengiriman.
- Mengklasifikasikan waktu pengiriman menjadi **Fast**, **Medium**, dan **Slow**.
- Memprediksi waktu pengiriman dalam menit.
- Mengevaluasi kedua model.
- Mengukur kontribusi fitur dan standardisasi melalui Ablation Study.

Penelitian ini menguji keterkaitan dan kontribusi fitur terhadap kinerja model, **bukan hubungan sebab-akibat**.

## 3. Dataset

[Food Delivery Time Prediction (Kaggle, denkuznetz)](https://www.kaggle.com/datasets/denkuznetz/food-delivery-time-prediction), file `Food_Delivery_Times.csv`.

| Kolom | Fungsi |
|---|---|
| Order_ID | Identitas pesanan (**tidak** dipakai sebagai fitur) |
| Distance_km | Fitur numerik |
| Weather, Traffic_Level, Time_of_Day, Vehicle_Type | Fitur kategorikal |
| Preparation_Time_min, Courier_Experience_yrs | Fitur numerik |
| Delivery_Time_min | Target (regresi) dan dasar pembentukan kategori (klasifikasi) |

Kategori klasifikasi (batas awal dari laporan, tidak diubah otomatis):

| Kategori | Waktu pengiriman |
|---|---|
| Fast | ≤ 30 menit |
| Medium | > 30 hingga 45 menit |
| Slow | > 45 menit |

Dataset tidak disertakan di repositori ini. Letakkan `Food_Delivery_Times.csv` di folder `data/` atau unggah lewat halaman **Upload Dataset**.

## 4. Algoritma

| Jalur | Algoritma | Keluaran | Standardisasi |
|---|---|---|---|
| 1 | `DecisionTreeClassifier` | Kategori Fast/Medium/Slow | Tidak |
| 2 | `KNeighborsRegressor` | Estimasi waktu pengiriman (menit) | Ya, hanya fitur numerik |

Parameter ditetapkan tetap dan sama di baseline serta seluruh skenario Ablation Study (tanpa tuning):

- Decision Tree: `criterion='gini'`, `max_depth=5`, `random_state=42`
- KNN: `n_neighbors=5`, `weights='uniform'`, `metric='minkowski'`, `p=2`

## 5. Struktur proyek

```
food-delivery-prediction/
├── app.py                  # titik masuk: halaman, tema, dan navigasi
├── requirements.txt
├── README.md
├── .streamlit/config.toml  # tema dasar Streamlit
├── data/                   # letakkan Food_Delivery_Times.csv di sini
├── src/                    # LOGIKA PENELITIAN (tidak berhubungan dengan tampilan)
│   ├── data_loader.py      # baca/validasi CSV, konstanta penelitian, target, split
│   ├── preprocessing.py    # Pipeline + ColumnTransformer
│   ├── eda.py              # statistik dan grafik EDA
│   ├── classification.py   # Decision Tree + metrik klasifikasi
│   ├── regression.py       # KNN Regressor + metrik regresi
│   ├── ablation.py         # 15 pengujian, tabel delta, interpretasi, Confusion Matrix
│   └── prediction.py       # pelatihan baseline dan prediksi pesanan baru
├── ui/                     # LAPISAN TAMPILAN (tidak berisi perhitungan ML)
│   ├── theme.py            # palet Light/Dark, CSS global, utilitas kontras
│   ├── markup.py           # pembuat HTML: kartu, tabel, callout, panel
│   ├── components.py       # pembungkus Streamlit untuk komponen
│   ├── charts.py           # grafik bertema Light/Dark
│   ├── content.py          # teks penjelasan metrik dan keterbatasan prediksi
│   └── pages.py            # 8 halaman dan sidebar
└── tests/
    ├── helpers.py          # data sintetis untuk test
    ├── test_preprocessing.py
    ├── test_models.py
    ├── test_ablation.py
    └── test_ui.py          # tema, kontras, HTML, grafik, teks penjelasan
```

## 6. Instalasi dan menjalankan (Windows)

Disarankan Python 3.10 atau lebih baru. Buka Command Prompt atau PowerShell di folder proyek:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Menjalankan unit test:

```bash
python -m pytest tests
```

## 7. Fitur aplikasi

| Halaman | Isi |
|---|---|
| Dashboard | Ringkasan dataset, kinerja baseline, penjelasan metrik evaluasi (Accuracy, MAE, RMSE, R²), keterbatasan penelitian |
| Upload Dataset | Unggah CSV, kecocokan kolom dengan laporan, tipe data, contoh data |
| Exploratory Data Analysis | Statistik deskriptif, distribusi, hubungan dengan waktu pengiriman, korelasi |
| Data Preprocessing | Missing values, duplikat, outlier, pembentukan target, distribusi kelas, split, missing sebelum/sesudah |
| Model Classification | Parameter, Accuracy, Precision/Recall/F1 per kelas dan macro, Confusion Matrix, Classification Report |
| Model Regression | Parameter, MAE, RMSE, R², tabel aktual vs prediksi |
| Ablation Study | 7 bagian: ringkasan eksperimen, hasil Decision Tree, hasil KNN, perbandingan dengan baseline, visualisasi, grid 7 Confusion Matrix, detail dan unduhan CSV |
| Prediction | Penjelasan perbedaan dan keterbatasan kedua model, formulir pesanan baru: kategori (Decision Tree) dan estimasi menit (KNN) |

## 8. Tampilan dan tema

Aplikasi memiliki **Light Mode** dan **Dark Mode**. Gunakan toggle **Mode gelap** di sidebar (bagian *Tampilan*). Pilihan tema tersimpan selama sesi dan ikut di URL (`?theme=dark`) sehingga bertahan saat halaman di-refresh; Anda juga dapat membuka aplikasi langsung dengan `?theme=dark`.

Catatan:

- Gunakan toggle di sidebar, bukan menu Settings bawaan Streamlit, supaya seluruh komponen kustom mengikuti palet yang sama.
- Font Inter dimuat dari Google Fonts; tanpa internet, aplikasi memakai font sistem.
- Tabel ditampilkan sebagai tabel HTML bertema (bukan `st.dataframe`) agar mengikuti tema; data yang ditampilkan sama.
- Tampilan hanya berada di folder `ui/`. Folder `src/` tidak berisi kode tampilan sehingga perubahan desain tidak memengaruhi hasil perhitungan.

## 9. Preprocessing

- **Missing values**: numerik diisi **median**, kategorikal diisi **modus**. Nilai dipelajari dari data training saja.
- **Target kosong**: bila `Delivery_Time_min` kosong, barisnya dikeluarkan dari pelatihan (target tidak diimputasi) dan jumlahnya dilaporkan.
- **Encoding**: `OneHotEncoder(handle_unknown='ignore')` untuk fitur kategorikal, selalu ada di semua skenario.
- **Standardisasi**: `StandardScaler` hanya untuk jalur KNN dan hanya pada fitur numerik; hasil One-Hot tidak distandardisasi.
- **Outlier**: hanya dilaporkan (IQR), tidak ada data yang dihapus.
- **Tanpa data leakage**: seluruh langkah di atas berada di dalam `Pipeline`/`ColumnTransformer` dan di-fit hanya pada data training. Data testing hanya ditransformasi dan dievaluasi.
- **Split**: 80:20, `random_state=42`, stratified berdasarkan kategori. Satu split dipakai oleh kedua model dan seluruh skenario Ablation Study.

## 10. Evaluasi model

**Decision Tree**: Accuracy, Precision, Recall, F1-score (per kelas dan macro average), Confusion Matrix, Classification Report. Macro average dipakai karena distribusi kelas tidak seimbang: setiap kelas berbobot sama. Aplikasi juga menampilkan Accuracy bila selalu menebak kelas terbanyak sebagai pembanding konteks.

**KNN**: MAE dan RMSE (menit, lebih rendah lebih baik) serta R² (lebih tinggi umumnya lebih baik).

## 11. Ablation Study

| Skenario | Perubahan | Model |
|---|---|---|
| Baseline | Seluruh fitur | Kedua model |
| Eksperimen 1 | Tanpa Distance_km | Kedua model |
| Eksperimen 2 | Tanpa Weather dan Traffic_Level | Kedua model |
| Eksperimen 3 | Tanpa Time_of_Day | Kedua model |
| Eksperimen 4 | Tanpa Vehicle_Type | Kedua model |
| Eksperimen 5 | Tanpa Preparation_Time_min | Kedua model |
| Eksperimen 6 | Tanpa Courier_Experience_yrs | Kedua model |
| Eksperimen 7 | Tanpa standardisasi fitur numerik | KNN saja |

Total **15 pengujian**: Decision Tree 7 (baseline + eksperimen 1-6) dan KNN 8 (baseline + eksperimen 1-7).

Aturan yang dijaga kode: indeks training-testing, random state, dan parameter model sama di semua skenario; pipeline dibuat baru dan di-fit ulang pada training sesuai fitur yang tersisa; encoding tidak pernah dihilangkan; target tidak berubah. Kode juga memeriksa bahwa jumlah pengujian tepat 15. Tabel hasil memuat nilai metrik dan selisih (Δ = skenario − baseline). Untuk Decision Tree, Confusion Matrix (data testing, urutan kelas Fast, Medium, Slow) ditampilkan pada baseline dan Eksperimen 1-6; Eksperimen 7 tidak memilikinya karena hanya menguji KNN. Semua angka berasal dari eksekusi model.

Cara membaca: penurunan Accuracy/F1/R² atau kenaikan MAE/RMSE terhadap baseline menunjukkan komponen yang dihilangkan **kemungkinan** berkontribusi pada eksperimen tersebut. Karena hanya satu split yang dipakai, selisih kecil dapat berasal dari variasi data. Interpretasi tidak menyimpulkan sebab-akibat.

## 12. Keterbatasan

Hanya variabel dalam dataset yang digunakan. Kondisi jalan, rute kurir, antrean restoran, dan lalu lintas real-time tidak tercakup. Estimasi pada halaman Prediction berdasarkan pola dataset dan bukan jaminan waktu pengiriman aktual.
