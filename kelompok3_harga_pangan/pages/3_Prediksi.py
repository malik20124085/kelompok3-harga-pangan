import streamlit as st

import importlib
import pandas as pd
import predict

try:
    import altair as alt
except ImportError:
    alt = None


predict = importlib.reload(predict)


def format_rupiah(value):
    return f"Rp {value:,.0f}"


def format_percent(value):
    return f"{value:.2f}%"


df = predict.load_price_data()
harga_terakhir = df["Harga"].iloc[-1]
tanggal_terakhir_data = df["Tanggal"].iloc[-1].normalize()
tanggal_terakhir = tanggal_terakhir_data.strftime("%d-%m-%Y")
today = pd.Timestamp.today().normalize()
today_label = today.strftime("%d-%m-%Y")

st.title("Prediksi Harga Cabai Rawit Merah")
st.caption("Prediksi harga untuk beberapa periode ke depan")

st.write(
    """
    Atur sendiri jumlah hari prediksi untuk melihat perkiraan harga, perubahan harga,
    dan indikator tren pada periode mendatang.
    """
)

filter_col1, filter_col2 = st.columns([2, 1])

with filter_col1:
    forecast_days = st.slider(
        "Jumlah hari ke depan",
        min_value=1,
        max_value=30,
        value=7,
        step=1,
        help="Geser untuk menentukan berapa hari ke depan yang ingin diprediksi.",
    )

with filter_col2:
    forecast_days = st.number_input(
        "Input manual",
        min_value=1,
        max_value=30,
        value=forecast_days,
        step=1,
        help="Masukkan jumlah hari secara manual.",
    )

forecast_days = int(forecast_days)
days_from_last_data_to_today = max((today - tanggal_terakhir_data).days, 0)
prediction_days_needed = days_from_last_data_to_today + forecast_days
raw_forecast_df = predict.predict_future_days(prediction_days_needed)

if days_from_last_data_to_today > 0:
    current_reference_row = raw_forecast_df[raw_forecast_df["Tanggal"] == today]
    current_reference_price = (
        current_reference_row["Prediksi Harga"].iloc[-1]
        if not current_reference_row.empty
        else harga_terakhir
    )
    reference_label = f"Estimasi hari ini ({today_label})"
    forecast_df = raw_forecast_df[raw_forecast_df["Tanggal"] > today].head(forecast_days)
else:
    current_reference_price = harga_terakhir
    reference_label = f"Harga terakhir ({tanggal_terakhir})"
    forecast_df = raw_forecast_df.head(forecast_days)

last_prediction = forecast_df["Prediksi Harga"].iloc[-1]
total_change = last_prediction - current_reference_price
total_change_percent = total_change / current_reference_price * 100
selected_prediction_date = forecast_df["Tanggal"].iloc[-1].strftime("%d-%m-%Y")

if total_change > 0:
    trend_label = "Naik"
elif total_change < 0:
    trend_label = "Turun"
else:
    trend_label = "Stabil"

st.divider()

st.subheader("Hasil Utama Prediksi")

highlight_col1, highlight_col2 = st.columns([2, 1])

with highlight_col1:
    st.info(
        f"""
        **Prediksi untuk {forecast_days} hari ke depan**

        Pada tanggal **{selected_prediction_date}**, harga Cabai Rawit Merah
        diprediksi menjadi **{format_rupiah(last_prediction)}**.
        """
    )

with highlight_col2:
    st.metric(
        f"Perubahan dari {reference_label}",
        format_rupiah(total_change),
        delta=format_percent(total_change_percent),
    )

st.divider()

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    st.metric("Referensi Hari Ini", format_rupiah(current_reference_price))
    st.caption(reference_label)

with metric_col2:
    st.metric(
        f"Prediksi Hari ke-{forecast_days}",
        format_rupiah(last_prediction),
        delta=format_rupiah(total_change),
    )

with metric_col3:
    st.metric("Perubahan Total", format_percent(total_change_percent))

with metric_col4:
    st.metric("Indikator Tren", trend_label)

st.divider()

if trend_label == "Naik":
    st.success(
        f"Selama {forecast_days} hari ke depan, harga diprediksi cenderung naik "
        f"sebesar {format_rupiah(total_change)}."
    )
elif trend_label == "Turun":
    st.warning(
        f"Selama {forecast_days} hari ke depan, harga diprediksi cenderung turun "
        f"sebesar {format_rupiah(abs(total_change))}."
    )
else:
    st.info(f"Selama {forecast_days} hari ke depan, harga diprediksi relatif stabil.")

st.subheader("Grafik Prediksi")

chart_df = forecast_df[["Tanggal", "Prediksi Harga"]].copy()
chart_df = chart_df.rename(columns={"Prediksi Harga": "Harga"})

if alt:
    chart = (
        alt.Chart(chart_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("Tanggal:T", title="Tanggal"),
            y=alt.Y("Harga:Q", title="Harga"),
            tooltip=[
                alt.Tooltip("Tanggal:T", title="Tanggal", format="%d-%m-%Y"),
                alt.Tooltip("Harga:Q", title="Harga", format=",.0f"),
            ],
        )
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.line_chart(chart_df.set_index("Tanggal"))

st.subheader("Detail Prediksi")

st.caption(
    "Tabel ini menampilkan rincian harian. Baris paling bawah adalah hasil utama "
    "sesuai jumlah hari yang dipilih."
)

display_df = forecast_df.copy()
display_df["Tanggal"] = display_df["Tanggal"].dt.strftime("%d-%m-%Y")
display_df["Prediksi Harga"] = display_df["Prediksi Harga"].map(format_rupiah)
display_df["Perubahan"] = display_df["Perubahan"].map(format_rupiah)
display_df["Perubahan (%)"] = display_df["Perubahan (%)"].map(format_percent)

st.dataframe(
    display_df.style.apply(
        lambda row: [
            "background-color: #fff3cd; font-weight: 700"
            if row.name == len(display_df) - 1
            else ""
            for _ in row
        ],
        axis=1,
    ),
    use_container_width=True,
    hide_index=True,
)

st.divider()

st.subheader("Informasi Model")

model_col1, model_col2, model_col3 = st.columns(3)

with model_col1:
    st.metric("Model", "Random Forest")

with model_col2:
    st.metric("MAPE", "2.58%")

with model_col3:
    st.metric("Jumlah Data", len(df))
