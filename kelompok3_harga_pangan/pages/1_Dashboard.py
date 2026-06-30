import pandas as pd
import streamlit as st

import importlib
import predict

try:
    import altair as alt
except ImportError:
    alt = None


predict = importlib.reload(predict)


def card_container():
    try:
        return st.container(border=True)
    except TypeError:
        return st.container()


df = pd.read_excel("data/processed/Data_Harga_Interpolasi_v2.xlsx")
df["Tanggal"] = pd.to_datetime(df["Tanggal"])
df = df.sort_values("Tanggal").reset_index(drop=True)

harga_terakhir = df["Harga"].iloc[-1]
harga_rata_rata = df["Harga"].mean()
harga_tertinggi = df["Harga"].max()
harga_terendah = df["Harga"].min()
tanggal_awal = df["Tanggal"].min().strftime("%d-%m-%Y")
tanggal_akhir = df["Tanggal"].max().strftime("%d-%m-%Y")
tanggal_terakhir_data = df["Tanggal"].max().normalize()
today = pd.Timestamp.today().normalize()
days_from_last_data_to_today = max((today - tanggal_terakhir_data).days, 0)
dashboard_forecast_days = 30
raw_forecast_df = predict.predict_future_days(days_from_last_data_to_today + dashboard_forecast_days)

st.title("Harga dan Prediksi Cabai Rawit Merah Jawa Barat")

st.header("Selamat Datang")
st.write(
    """
    Aplikasi ini membantu melihat perkembangan harga Cabai Rawit Merah di Jawa Barat
    dan memberikan prediksi harga untuk periode berikutnya menggunakan model machine learning.
    """
)

st.divider()

st.subheader("Ringkasan Dataset")

price_col1, price_col2, price_col3, price_col4 = st.columns(4)

with price_col1:
    st.metric("Jumlah Data", len(df))

with price_col2:
    st.metric("Harga Terakhir", f"Rp {harga_terakhir:,.0f}")

with price_col3:
    st.metric("Rata-rata Harga", f"Rp {harga_rata_rata:,.0f}")

with price_col4:
    st.metric("Harga Tertinggi", f"Rp {harga_tertinggi:,.0f}")

st.caption(f"Harga terendah dalam dataset: Rp {harga_terendah:,.0f}")

st.divider()

st.subheader("Grafik Harga Historis dan Prediksi")

historical_chart_df = df[["Tanggal", "Harga"]].copy()
historical_chart_df = historical_chart_df.rename(columns={"Harga": "Harga Aktual"})
historical_chart_df["Prediksi Harga"] = float("nan")

forecast_chart_df = pd.concat(
    [
        pd.DataFrame(
            [
                {
                    "Tanggal": tanggal_terakhir_data,
                    "Prediksi Harga": harga_terakhir,
                }
            ]
        ),
        raw_forecast_df[["Tanggal", "Prediksi Harga"]],
    ],
    ignore_index=True,
)
forecast_chart_df["Harga Aktual"] = float("nan")

chart_df = pd.concat(
    [
        historical_chart_df[["Tanggal", "Harga Aktual", "Prediksi Harga"]],
        forecast_chart_df[["Tanggal", "Harga Aktual", "Prediksi Harga"]],
    ],
    ignore_index=True,
)

if alt:
    chart_long_df = chart_df.melt(
        id_vars="Tanggal",
        var_name="Jenis Data",
        value_name="Harga",
    ).dropna(subset=["Harga"])

    chart = (
        alt.Chart(chart_long_df)
        .mark_line()
        .encode(
            x=alt.X("Tanggal:T", title="Tanggal"),
            y=alt.Y("Harga:Q", title="Harga"),
            color=alt.Color("Jenis Data:N", title="Jenis Data"),
            tooltip=[
                alt.Tooltip("Tanggal:T", title="Tanggal", format="%d-%m-%Y"),
                alt.Tooltip("Jenis Data:N", title="Jenis Data"),
                alt.Tooltip("Harga:Q", title="Harga", format=",.0f"),
            ],
        )
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.line_chart(chart_df.set_index("Tanggal"))

st.caption(
    "Grafik menampilkan seluruh data historis dari awal dataset, lalu dilanjutkan "
    "dengan garis prediksi sampai hari ini dan proyeksi 30 hari ke depan."
)

st.divider()

about_col, nav_col = st.columns(2)

with about_col:
    st.subheader("Tentang Dataset")
    st.write(
        f"""
        Dataset yang digunakan berisi data historis harga Cabai Rawit Merah di Jawa Barat.
        Data memiliki kolom utama **Tanggal** dan **Harga**, dengan rentang periode
        dari **{tanggal_awal}** sampai **{tanggal_akhir}**.

        Data ini sudah melalui proses interpolasi sehingga dapat digunakan untuk analisis
        tren harga dan pembuatan fitur prediksi seperti lag, moving average, dan statistik
        rolling.
        """
    )

with nav_col:
    st.subheader("Navigasi Aplikasi")
    st.write("Pilih halaman yang ingin dibuka. Setiap tombol di bawah bisa langsung diklik.")

    nav_pages = [
        {
            "title": "EDA",
            "description": "Eksplorasi data dan pola perubahan harga.",
            "path": "kelompok3_harga_pangan/pages/2_EDA.py",
        },
        {
            "title": "Prediksi",
            "description": "Lihat hasil prediksi harga periode berikutnya.",
            "path": "kelompok3_harga_pangan/pages/3_Prediksi.py",
        },
        {
            "title": "Tentang Model",
            "description": "Pelajari model machine learning yang digunakan.",
            "path": "kelompok3_harga_pangan/pages/4_Tentang_Model.py",
        },
    ]

    for page in nav_pages:
        with card_container():
            st.markdown(f"**{page['title']}**")
            st.caption(page["description"])

            if hasattr(st, "page_link"):
                st.page_link(
                    page["path"],
                    label=f"Klik untuk buka {page['title']}",
                    help=page["description"],
                )
            elif hasattr(st, "switch_page"):
                if st.button(f"Klik untuk buka {page['title']}"):
                    st.switch_page(page["path"])
            else:
                st.caption("Buka halaman ini melalui menu sidebar.")
