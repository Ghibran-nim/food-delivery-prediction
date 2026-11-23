"""Unit test lapisan UI (tema, HTML, grafik). Tidak membutuhkan Streamlit."""
import numpy as np
import pandas as pd

from tests.helpers import make_synthetic_df  # noqa: F401  (juga mengatur sys.path)
from src import ablation as abl
from src import data_loader as dl
from src import prediction as pred
from ui import charts, content, markup, theme

_CACHE = {}


def _ctx():
    if "data" not in _CACHE:
        data = dl.build_experiment_data(make_synthetic_df(n=500, seed=3))
        _CACHE["data"] = data
        _CACHE["models"] = pred.train_baseline_models(data)
        _CACHE["ablation"] = abl.run_ablation(data)
    return _CACHE["data"], _CACHE["models"], _CACHE["ablation"]


# --- Sistem tema ------------------------------------------------------------
SPEC_COLORS = {
    "light": {"bg": "#F8FAFC", "sidebar": "#FFFFFF", "card": "#FFFFFF", "primary": "#2563EB", "secondary": "#0D9488",
              "text": "#0F172A", "muted": "#64748B", "border": "#E2E8F0", "success": "#16A34A",
              "warning": "#D97706", "danger": "#DC2626"},
    "dark": {"bg": "#0B1120", "sidebar": "#111827", "card": "#172033", "primary": "#60A5FA", "secondary": "#2DD4BF",
             "text": "#F1F5F9", "muted": "#94A3B8", "border": "#334155", "success": "#4ADE80",
             "warning": "#FBBF24", "danger": "#F87171"},
}


def test_palettes_match_design_specification():
    for mode_name, colors in SPEC_COLORS.items():
        for key, value in colors.items():
            assert theme.PALETTES[mode_name][key] == value, (mode_name, key)


def test_resolve_theme_defaults_to_light():
    assert theme.resolve_theme("dark") == "dark" and theme.resolve_theme("DARK") == "dark"
    assert theme.resolve_theme("light") == "light" and theme.resolve_theme(None) == "light"
    assert theme.resolve_theme("apa-saja") == "light"


def test_css_uses_active_theme_colors_and_leaves_no_placeholders():
    for mode_name in ("light", "dark"):
        css = theme.build_css(mode_name)
        assert "${" not in css and "$" not in css
        assert theme.PALETTES[mode_name]["bg"] in css and theme.PALETTES[mode_name]["card"] in css
        assert f"color-scheme: {mode_name}" in css
    assert theme.PALETTES["dark"]["bg"] not in theme.build_css("light")
    assert theme.PALETTES["light"]["bg"] not in theme.build_css("dark")


def test_contrast_of_text_pairs_meets_wcag_aa_for_normal_text():
    """Pasangan teks utama dan teks pada tombol/tab/tabel harus minimal 4.5:1."""
    for mode_name in ("light", "dark"):
        t = theme.tokens(mode_name)
        pairs = {
            "text/bg": (t["text"], t["bg"]), "text/card": (t["text"], t["card"]),
            "text/sidebar": (t["text"], t["sidebar"]), "text/hover": (t["text"], t["hover"]),
            "text/table_head": (t["text"], t["table_head"]),
            "on_primary/primary (tombol)": (t["on_primary"], t["primary"]),
            "primary/card (tab aktif, tautan)": (t["primary"], t["card"]),
            "text/tint_primary (menu sidebar aktif)": (t["text"], t["tint_primary"]),
        }
        for name, (fg, bg) in pairs.items():
            assert theme.contrast_ratio(fg, bg) >= 4.5, (mode_name, name, round(theme.contrast_ratio(fg, bg), 2))
        for kind in ("primary", "success", "warning", "danger"):   # callout: teks utama di atas warna tint
            assert theme.contrast_ratio(t["text"], t[f"tint_{kind}"]) >= 4.5, (mode_name, kind)


def test_secondary_text_contrast_is_reported_not_hidden():
    """Teks sekunder (warna spesifikasi) dicatat: harus minimal 3:1 (ambang teks besar/komponen UI)."""
    for mode_name in ("light", "dark"):
        t = theme.tokens(mode_name)
        for surface in ("bg", "card"):
            assert theme.contrast_ratio(t["muted"], t[surface]) >= 3.0, (mode_name, surface)


def test_contrast_function_known_values():
    assert abs(theme.contrast_ratio("#000000", "#FFFFFF") - 21.0) < 1e-6
    assert abs(theme.contrast_ratio("#777777", "#777777") - 1.0) < 1e-9


# --- Komponen HTML ------------------------------------------------------------
def test_table_html_escapes_untrusted_text_and_shows_nan():
    df = pd.DataFrame({"Nama": ["<script>alert(1)</script>", "b"], "Nilai": [1.5, np.nan]})
    html_out = markup.table_html(df)
    assert "<script>" not in html_out and "&lt;script&gt;" in html_out
    assert 'class="fd-na">NaN' in html_out
    assert "\n" not in html_out                                   # satu baris: aman dari parser Markdown


def test_table_html_values_match_dataframe():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [0.12345678, 2.0, 3.5], "C": ["x", "y", "z"]})
    out = markup.table_html(df, decimals=4)
    for text in ["0.1235", "2.0000", "3.5000", ">x<", ">y<", ">z<"]:
        assert text in out, text
    assert out.count("<tr>") == len(df) + 1                        # header + baris data


def test_table_html_delta_arrows_follow_metric_direction():
    df = pd.DataFrame({"Δ Accuracy": [-0.05, 0.02, 0.0], "Δ MAE": [0.5, -0.5, 0.0]})
    out = markup.table_html(df, delta_cols={"Δ Accuracy": True, "Δ MAE": False})
    rows = out.split("<tr>")[2:]
    # baris 1: Accuracy turun (buruk), MAE naik (buruk)
    assert rows[0].count("fd-arrow-bad") == 2 and "&#9660;" in rows[0] and "&#9650;" in rows[0]
    # baris 2: Accuracy naik (baik), MAE turun (baik)
    assert rows[1].count("fd-arrow-good") == 2
    # baris 3: tidak berubah
    assert rows[2].count("fd-arrow-flat") == 2 and "-0.0500" in rows[0] and "+0.0200" in rows[1]


def test_callout_cards_and_panels_escape_text():
    assert "&lt;b&gt;" in markup.callout_html("info", "<b>x</b>")
    assert "<b>x</b>" in markup.callout_html("info", "<b>x</b>", allow_html=True)
    assert "fd-callout warning" in markup.callout_html("warning", "x")
    assert "fd-callout danger" in markup.callout_html("danger", "x")
    cards = markup.cards_html([{"label": "<L>", "value": "1", "accent": True, "hint": "h"}])
    assert "&lt;L&gt;" in cards and "fd-card-accent" in cards and "\n" not in cards
    assert "FOOD DELIVERY ANALYTICS" in markup.brand_html() and "Machine Learning Prediction Platform" in markup.brand_html()


# --- Grafik: nilai yang digambar = nilai pada tabel -----------------------------
def _heatmap_values(ax):
    return np.asarray(ax.collections[0].get_array()).astype(int)


def test_confusion_matrix_figure_draws_exact_cell_values_in_class_order():
    import matplotlib.pyplot as plt
    _, models, _ = _ctx()
    for mode_name in ("light", "dark"):
        fig = charts.confusion_matrix_figure(models.clf.confusion, mode=mode_name)
        ax = fig.axes[0]
        assert (_heatmap_values(ax).reshape(3, 3) == models.clf.confusion.values).all()
        assert [t.get_text() for t in ax.get_xticklabels()] == dl.CLASS_ORDER
        assert [t.get_text() for t in ax.get_yticklabels()] == dl.CLASS_ORDER
        plt.close(fig)


def test_confusion_grid_has_seven_matrices_with_exact_values_and_order():
    import matplotlib.pyplot as plt
    _, _, res = _ctx()
    for mode_name in ("light", "dark"):
        fig = charts.confusion_grid(res.confusions, res.clf_table, mode=mode_name)
        panels = [ax for ax in fig.axes if ax.collections]
        titles = [ax.get_title(loc="left") for ax in panels]
        assert len(panels) == 7 and "Eksperimen 7" not in " ".join(titles)
        for ax, (name, cm) in zip(panels, res.confusions.items()):
            assert name in ax.get_title(loc="left")
            assert (_heatmap_values(ax).reshape(3, 3) == cm.values).all(), name
            assert [t.get_text() for t in ax.get_xticklabels()] == dl.CLASS_ORDER
        plt.close(fig)


def test_ablation_charts_draw_exact_table_values():
    import matplotlib.pyplot as plt
    _, _, res = _ctx()
    fig = charts.ablation_classification(res.clf_table, mode="dark")
    for ax, metric in zip(fig.axes, charts.CLF_METRICS):
        heights = [p.get_height() for p in ax.patches]
        assert np.allclose(heights, res.clf_table[metric].values), metric
        assert len(heights) == 7
    plt.close(fig)
    fig = charts.ablation_regression(res.reg_table, mode="light")
    for ax, metric in zip(fig.axes, charts.REG_METRICS):
        heights = [p.get_height() for p in ax.patches]
        assert np.allclose(heights, res.reg_table[metric].values), metric
        assert len(heights) == 8
    plt.close(fig)


def test_actual_vs_predicted_uses_model_outputs():
    import matplotlib.pyplot as plt
    _, models, _ = _ctx()
    fig = charts.actual_vs_predicted(models.reg.y_true, models.reg.y_pred, mode="dark")
    offsets = np.asarray(fig.axes[0].collections[0].get_offsets())
    assert np.allclose(offsets[:, 0], np.asarray(models.reg.y_true, dtype=float))
    assert np.allclose(offsets[:, 1], np.asarray(models.reg.y_pred, dtype=float))
    plt.close(fig)


def test_chart_backgrounds_follow_theme():
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex
    data, _, _ = _ctx()
    for mode_name in ("light", "dark"):
        fig = charts.histogram(data.df, dl.TARGET_COL, "t", "x", mode=mode_name)
        assert to_hex(fig.get_facecolor()).upper() == theme.PALETTES[mode_name]["card"].upper()
        assert to_hex(fig.axes[0].get_facecolor()).upper() == theme.PALETTES[mode_name]["card"].upper()
        plt.close(fig)


def test_all_eda_charts_render_in_both_themes():
    import matplotlib.pyplot as plt
    from src import eda
    data, _, _ = _ctx()
    df = data.df
    for m in ("light", "dark"):
        figs = [
            charts.histogram(df, "Distance_km", "t", "x", mode=m),
            charts.countplot(df, "Weather", "t", mode=m),
            charts.distance_vs_delivery(df, mode=m),
            charts.box_by_category(df, "Traffic_Level", "t", mode=m),
            charts.correlation_heatmap(eda.numeric_correlation(df), mode=m),
            charts.class_distribution(dl.class_distribution_all(data), mode=m),
        ]
        assert all(f is not None for f in figs)
        for f in figs:
            plt.close(f)


def test_heatmap_annotation_contrast_meets_aa_in_both_themes():
    """Angka pada setiap sel Confusion Matrix dan korelasi harus terbaca (kontras >= 4.5:1 terhadap warna sel)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex
    _, models, res = _ctx()
    spread = pd.DataFrame([[0, 7, 30], [3, 60, 11], [1, 8, 200]], index=models.clf.confusion.index,
                          columns=models.clf.confusion.columns)       # rentang nilai lebar: sel pucat sampai pekat
    corr = pd.DataFrame(np.linspace(-1, 1, 16).reshape(4, 4), index=list("abcd"), columns=list("abcd"))
    for mode_name in ("light", "dark"):
        figs = [charts.confusion_matrix_figure(spread, mode=mode_name),
                charts.confusion_grid(res.confusions, res.clf_table, mode=mode_name),
                charts.correlation_heatmap(corr, mode=mode_name)]
        for fig in figs:
            for ax in [a for a in fig.axes if a.collections]:
                mesh = ax.collections[0]
                mesh.update_scalarmappable()
                for text_artist, rgba in zip(ax.texts, mesh.get_facecolor()):
                    ratio = theme.contrast_ratio(to_hex(text_artist.get_color()), to_hex(rgba))
                    assert ratio >= 4.5, (mode_name, text_artist.get_text(), round(ratio, 2))
            plt.close(fig)


# --- Penjelasan metrik (Dashboard) dan keterbatasan prediksi (Prediction) ------------
def test_fmt_id_uses_decimal_comma_and_conventional_rounding():
    assert content.fmt_id(8.735, 2) == "8,74"          # setengah ke atas, bukan artefak biner
    assert content.fmt_id(84.5, 1) == "84,5" and content.fmt_id(0.845 * 100, 1) == "84,5"
    assert content.fmt_id(11.996, 2) == "12,00" and content.fmt_id(0.6902, 2) == "0,69"
    assert content.fmt_id(69.02, 0) == "69" and content.fmt_id(-0.1234, 2) == "-0,12"


def test_metric_groups_cover_four_metrics_grouped_by_task():
    groups = content.metric_groups()
    assert [g["task"] for g in groups] == ["Classification - Decision Tree", "Regression - KNN Regressor"]
    titles = [[m["title"] for m in g["metrics"]] for g in groups]
    assert titles[0] == ["Accuracy"]
    assert titles[1] == ["MAE (Mean Absolute Error)", "RMSE (Root Mean Squared Error)", "R² (R-Squared)"]
    by_title = {m["title"]: m for g in groups for m in g["metrics"]}
    assert "persentase prediksi kategori yang benar" in by_title["Accuracy"]["definition"].lower()
    assert "rata-rata selisih absolut" in by_title["MAE (Mean Absolute Error)"]["definition"].lower()
    assert "penalti lebih besar" in by_title["RMSE (Root Mean Squared Error)"]["definition"].lower()
    r2 = by_title["R² (R-Squared)"]
    assert "variasi waktu pengiriman" in r2["definition"] and "bukan persentase prediksi yang benar" in r2["definition"]
    assert "mendekati 1" in r2["direction"]
    for name in ("MAE (Mean Absolute Error)", "RMSE (Root Mean Squared Error)"):
        assert by_title[name]["direction"] == "Semakin kecil, semakin baik"
    assert by_title["Accuracy"]["direction"] == "Semakin tinggi, semakin baik"


def test_metric_examples_use_actual_values_when_models_exist():
    _, models, _ = _ctx()
    by_title = {m["title"]: m for g in content.metric_groups(models.clf.summary, models.reg.summary) for m in g["metrics"]}
    acc = content.fmt_id(models.clf.summary["Accuracy"] * 100, 1)
    assert f"Accuracy {acc}%" in by_title["Accuracy"]["example"] and by_title["Accuracy"]["example"].startswith("Pada hasil saat ini")
    mae = content.fmt_id(models.reg.summary["MAE"], 2)
    assert f"MAE {mae} menit" in by_title["MAE (Mean Absolute Error)"]["example"]
    rmse = content.fmt_id(models.reg.summary["RMSE"], 2)
    assert f"RMSE {rmse} menit" in by_title["RMSE (Root Mean Squared Error)"]["example"]
    r2 = content.fmt_id(models.reg.summary["R²"], 2)
    assert f"R² {r2}" in by_title["R² (R-Squared)"]["example"]


def test_metric_examples_without_dataset_are_labelled_illustrative():
    for g in content.metric_groups():
        for m in g["metrics"]:
            assert m["example"].startswith("Misalnya:"), m["title"]


def test_negative_r2_is_not_described_as_explained_variance():
    groups = content.metric_groups({"Accuracy": 0.5}, {"MAE": 5.0, "RMSE": 6.0, "R²": -0.2})
    r2_example = [m for g in groups for m in g["metrics"] if m["title"].startswith("R²")][0]["example"]
    assert "negatif" in r2_example and "variasi waktu pengiriman pada data pengujian" not in r2_example


def test_prediction_note_covers_required_points_in_plain_language():
    items = content.prediction_note_items()
    text = " ".join(h + " " + t for h, t in items)
    assert len(items) == 3 and len(text) < 1100                        # ringkas
    for phrase in ("Decision Tree", "KNN", "kategori", "angka dalam menit", "sendiri-sendiri",
                   "tidak sepenuhnya sesuai", "bukan otomatis berarti aplikasi salah", "bukan jaminan", "pola data pelatihan"):
        assert phrase in text, phrase
    assert f"{dl.FAST_MAX}" in text and f"{dl.MEDIUM_MAX}" in text      # batas kategori berasal dari konstanta inti


def test_consistency_note_only_when_two_results_disagree():
    assert content.consistency_note("Slow", 50.0) is None
    assert content.consistency_note("Fast", 10.0) is None
    assert content.consistency_note("Medium", 45.0) is None and content.consistency_note("Slow", 45.0) is not None
    assert content.consistency_note("Medium", 30.0) is not None and content.consistency_note("Fast", 30.0) is None
    note = content.consistency_note("Slow", 40.0)
    assert "Slow" in note and "Medium" in note and "40,0 menit" in note and "bukan otomatis" in note
    assert content.consistency_note("Slow", float("nan")) is None


def test_consistency_note_does_not_modify_predictions():
    _, models, _ = _ctx()
    order = {"Distance_km": 8.5, "Weather": "Rainy", "Traffic_Level": "High", "Time_of_Day": "Evening",
             "Vehicle_Type": "Bike", "Preparation_Time_min": 15, "Courier_Experience_yrs": 3.0}
    before = pred.predict_new_order(models, order)
    content.consistency_note(before["kategori"], before["menit"])
    assert pred.predict_new_order(models, order) == before


def test_explain_panel_and_note_list_markup_is_safe_single_line_html():
    panel = markup.explain_panel_html("<MAE>", "def <b>x</b>", "contoh", "Semakin kecil, semakin baik")
    assert "&lt;MAE&gt;" in panel and "<b>x</b>" not in panel and "\n" not in panel and "fd-badge primary" in panel
    lst = markup.note_list_html([("Judul.", "isi <i>x</i>")])
    assert lst.startswith('<ul class="fd-list">') and "&lt;i&gt;" in lst and "\n" not in lst
    for mode_name in ("light", "dark"):
        css = theme.build_css(mode_name)
        assert ".fd-text" in css and ".fd-list" in css and "$" not in css


def test_text_contrast_for_new_explanation_styles():
    for mode_name in ("light", "dark"):
        t = theme.tokens(mode_name)
        assert theme.contrast_ratio(t["text"], t["card"]) >= 4.5            # .fd-text pada panel
        assert theme.contrast_ratio(t["muted"], t["card"]) >= 4.5           # .fd-text-muted (contoh) pada panel
        assert theme.contrast_ratio(t["text"], t["tint_primary"]) >= 4.5    # daftar pada callout info
