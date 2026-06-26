import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    import altair as alt
except ImportError:
    alt = None


MODEL_PATH = "models/model_random_forest.pkl"
METRICS_PATH = "models/model_metrics.pkl"
SCALER_Y_PATH = "models/scaler_y.pkl"

FEATURES = [
    "Day",
    "Month",
    "Year",
    "DayOfWeek",
    "Quarter",
    "Lag_1",
    "Lag_3",
    "Lag_7",
    "Lag_14",
    "Lag_30",
    "MA_7",
    "MA_30",
    "Std_7",
]

FEATURE_DESCRIPTIONS = {
    "Day": ("Tanggal dalam bulan", "Hari"),
    "Month": ("Bulan dari tanggal data", "Bulan"),
    "Year": ("Tahun dari tanggal data", "Tahun"),
    "DayOfWeek": ("Urutan hari dalam minggu, Senin bernilai 0", "Hari"),
    "Quarter": ("Kuartal kalender dari tanggal data", "Kuartal"),
    "Lag_1": ("Harga 1 hari sebelumnya", "Rupiah"),
    "Lag_3": ("Harga 3 hari sebelumnya", "Rupiah"),
    "Lag_7": ("Harga 7 hari sebelumnya", "Rupiah"),
    "Lag_14": ("Harga 14 hari sebelumnya", "Rupiah"),
    "Lag_30": ("Harga 30 hari sebelumnya", "Rupiah"),
    "MA_7": ("Rata-rata bergerak harga 7 hari", "Rupiah"),
    "MA_30": ("Rata-rata bergerak harga 30 hari", "Rupiah"),
    "Std_7": ("Volatilitas harga 7 hari", "Rupiah"),
}


st.set_page_config(
    page_title="Tentang Model",
    page_icon=":robot_face:",
    layout="wide",
)


@st.cache_resource
def load_model_assets():
    metrics = joblib.load(METRICS_PATH)

    model = None
    scaler_y = None
    load_errors = []

    try:
        model = joblib.load(MODEL_PATH)
    except Exception as error:
        load_errors.append(f"Model tidak dapat dimuat: {error}")

    try:
        scaler_y = joblib.load(SCALER_Y_PATH)
    except Exception as error:
        load_errors.append(f"Scaler target tidak dapat dimuat: {error}")

    return model, metrics, scaler_y, load_errors


def rupiah(value: float) -> str:
    return f"Rp {value:,.0f}"


def inverse_target(scaler_y, values):
    values_array = np.asarray(values).reshape(-1, 1)
    if scaler_y is None:
        return values_array.ravel()
    return scaler_y.inverse_transform(values_array).ravel()


def metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="config-card">
            <span>{label}</span>
            <strong>{value}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <style>
    .config-card {
        background: #e8f2ff;
        border-radius: 8px;
        padding: 16px 18px;
        min-height: 74px;
    }

    .config-card span {
        color: #1f6fb2;
        display: block;
        font-size: 0.88rem;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .config-card strong {
        color: #1f2937;
        font-size: 1rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

model, metrics, scaler_y, load_errors = load_model_assets()
params = model.get_params() if model else {}

y_test_scaled = metrics["y_test"]
y_pred_scaled = metrics["y_pred"]
y_test = inverse_target(scaler_y, y_test_scaled)
y_pred = inverse_target(scaler_y, y_pred_scaled)

mae_rupiah = np.mean(np.abs(y_test - y_pred))
rmse_rupiah = np.sqrt(np.mean((y_test - y_pred) ** 2))
r2_score = metrics["r2"]
mape = metrics["mape"]
test_size = len(y_test)
train_size = int(round(test_size / 0.2 * 0.8))
total_samples = train_size + test_size

st.title("🤖 Tentang Model")
st.write(
    """
    Halaman ini menampilkan konfigurasi dan performa **Random Forest Regressor**
    yang digunakan untuk memprediksi harga Cabai Rawit Merah Jawa Barat.
    """
)

if load_errors:
    with st.expander("Catatan pemuatan aset model"):
        for load_error in load_errors:
            st.warning(load_error)
        st.caption(
            "Install dependency dari requirements.txt agar model dan scaler dapat "
            "dimuat penuh."
        )

st.subheader("⚙️ Konfigurasi Training")

config_row_1 = st.columns(3)
with config_row_1[0]:
    metric_card("Algoritma", "Random Forest Regressor")
with config_row_1[1]:
    metric_card("max_depth", str(params.get("max_depth", 10)))
with config_row_1[2]:
    metric_card("Training size", f"{train_size:,} sampel (80%)")

config_row_2 = st.columns(3)
with config_row_2[0]:
    metric_card("n_estimators", f"{params.get('n_estimators', 200)} pohon")
with config_row_2[1]:
    metric_card("random_state", str(params.get("random_state", 42)))
with config_row_2[2]:
    metric_card("Test size", f"{test_size:,} sampel (20%)")

st.divider()

st.subheader("📈 Performa Model")

perf_col1, perf_col2, perf_col3 = st.columns(3)

with perf_col1:
    st.metric("MAE (Test)", rupiah(mae_rupiah) if scaler_y else f"{mae_rupiah:.4f}")
    st.caption(f"Skala normalisasi: {metrics['mae']:.4f}")

with perf_col2:
    st.metric("RMSE (Test)", rupiah(rmse_rupiah) if scaler_y else f"{rmse_rupiah:.4f}")
    st.caption(f"Skala normalisasi: {metrics['rmse']:.4f}")

with perf_col3:
    st.metric("R² Score (Test)", f"{r2_score:.4f}")
    st.caption(f"MAPE: {mape:.2f}%")

interpretation_col1, interpretation_col2 = st.columns(2)

with interpretation_col1:
    if r2_score >= 0.8:
        st.success(f"R² = {r2_score:.3f} — Model menjelaskan pola harga dengan sangat baik.")
    elif r2_score >= 0.6:
        st.warning(f"R² = {r2_score:.3f} — Model cukup baik, tetapi masih dapat ditingkatkan.")
    else:
        st.error(f"R² = {r2_score:.3f} — Model belum menangkap pola data dengan baik.")

with interpretation_col2:
    st.info(
        "Rata-rata kesalahan absolut prediksi adalah sekitar "
        f"**{rupiah(mae_rupiah) if scaler_y else f'{mae_rupiah:.4f}'}** "
        "pada data uji."
    )

st.divider()

st.subheader("🎯 Aktual vs Prediksi (Test Set)")

prediction_df = pd.DataFrame(
    {
        "Harga Aktual": y_test,
        "Harga Prediksi": y_pred,
    }
)

min_price = float(prediction_df.min().min())
max_price = float(prediction_df.max().max())
perfect_line_df = pd.DataFrame(
    {
        "Harga Aktual": [min_price, max_price],
        "Harga Prediksi": [min_price, max_price],
        "Jenis": ["Prediksi sempurna", "Prediksi sempurna"],
    }
)

if alt:
    scatter = (
        alt.Chart(prediction_df)
        .mark_circle(size=46, opacity=0.55, color="#4aa3df")
        .encode(
            x=alt.X("Harga Aktual:Q", title="Harga Aktual"),
            y=alt.Y("Harga Prediksi:Q", title="Harga Prediksi"),
            tooltip=[
                alt.Tooltip("Harga Aktual:Q", title="Aktual", format=",.0f"),
                alt.Tooltip("Harga Prediksi:Q", title="Prediksi", format=",.0f"),
            ],
        )
    )
    perfect_line = (
        alt.Chart(perfect_line_df)
        .mark_line(color="red", strokeDash=[8, 6], strokeWidth=2)
        .encode(
            x="Harga Aktual:Q",
            y="Harga Prediksi:Q",
        )
    )
    st.altair_chart(
        (scatter + perfect_line)
        .properties(
            height=560,
            title=(
                f"Aktual vs Prediksi — R² = {r2_score:.4f} | "
                f"RMSE = {rupiah(rmse_rupiah) if scaler_y else f'{rmse_rupiah:.4f}'}"
            ),
        )
        .interactive(),
        use_container_width=True,
    )
else:
    st.scatter_chart(prediction_df)

st.caption(
    "Titik yang semakin dekat dengan garis merah putus-putus menunjukkan prediksi "
    "yang semakin dekat dengan harga aktual."
)

st.divider()

st.subheader("📊 Feature Importance")
st.write(
    "Feature importance menunjukkan kontribusi relatif setiap fitur dalam pengambilan "
    "keputusan model Random Forest."
)

if model:
    importance_values = model.feature_importances_
else:
    importance_values = np.zeros(len(FEATURES))

importance_df = pd.DataFrame(
    {
        "Fitur": FEATURES,
        "Importance": importance_values,
    }
).sort_values("Importance", ascending=False)
mean_importance = importance_df["Importance"].mean()

if alt:
    importance_chart = (
        alt.Chart(importance_df)
        .mark_bar(color="#3498db")
        .encode(
            x=alt.X("Importance:Q", title="Feature Importance"),
            y=alt.Y("Fitur:N", title="Fitur", sort="-x"),
            tooltip=[
                alt.Tooltip("Fitur:N", title="Fitur"),
                alt.Tooltip("Importance:Q", title="Importance", format=".3f"),
            ],
        )
    )
    mean_line = (
        alt.Chart(pd.DataFrame({"Importance": [mean_importance]}))
        .mark_rule(color="red", strokeDash=[7, 5], strokeWidth=2)
        .encode(x="Importance:Q")
    )
    st.altair_chart(
        (importance_chart + mean_line).properties(height=520),
        use_container_width=True,
    )
else:
    st.bar_chart(importance_df.set_index("Fitur")["Importance"])

if model:
    top_feature = importance_df.iloc[0]
    st.success(
        f"Fitur paling berpengaruh adalah **{top_feature['Fitur']}** "
        f"dengan importance **{top_feature['Importance']:.3f}**."
    )
else:
    st.info("Feature importance akan tampil setelah model berhasil dimuat.")

st.divider()

st.subheader("📋 Deskripsi Fitur Dataset")

feature_description_df = importance_df.copy()
feature_description_df["Deskripsi Lengkap"] = feature_description_df["Fitur"].map(
    lambda feature: FEATURE_DESCRIPTIONS[feature][0]
)
feature_description_df["Satuan"] = feature_description_df["Fitur"].map(
    lambda feature: FEATURE_DESCRIPTIONS[feature][1]
)
feature_description_df = feature_description_df[
    ["Fitur", "Deskripsi Lengkap", "Satuan", "Importance"]
]

st.dataframe(
    feature_description_df.style.format({"Importance": "{:.3f}"}),
    hide_index=True,
    use_container_width=True,
)

st.caption(
    f"Total data evaluasi: {total_samples:,} sampel setelah pembuatan fitur lag dan rolling."
)
