from pathlib import Path

import pandas as pd
import streamlit as st

try:
    import altair as alt
except ImportError:
    alt = None


DATA_PATH = Path("data/processed/Data_Harga_Interpolasi_v2.xlsx")
DATE_COLUMN = "Tanggal"
PRICE_COLUMN = "Harga"
COMMODITY_COLUMN = "Komoditas (Rp)"


st.set_page_config(
    page_title="EDA",
    page_icon=":bar_chart:",
    layout="wide",
)


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()

    required_columns = {DATE_COLUMN, PRICE_COLUMN}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        st.error(f"Kolom wajib tidak ditemukan: {missing_text}")
        st.stop()

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df[PRICE_COLUMN] = pd.to_numeric(df[PRICE_COLUMN], errors="coerce")

    return df.sort_values(DATE_COLUMN).reset_index(drop=True)


def rupiah(value: float) -> str:
    return f"Rp {value:,.0f}"


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    featured_df = df.copy()
    featured_df["Tahun"] = featured_df[DATE_COLUMN].dt.year
    featured_df["Bulan_Angka"] = featured_df[DATE_COLUMN].dt.month
    featured_df["Bulan"] = featured_df[DATE_COLUMN].dt.month_name()
    featured_df["Hari"] = featured_df[DATE_COLUMN].dt.day_name()
    featured_df["MA_7"] = featured_df[PRICE_COLUMN].rolling(window=7).mean()
    featured_df["MA_30"] = featured_df[PRICE_COLUMN].rolling(window=30).mean()
    featured_df["Std_7"] = featured_df[PRICE_COLUMN].rolling(window=7).std()

    for lag in [1, 3, 7, 14, 30]:
        featured_df[f"Lag_{lag}"] = featured_df[PRICE_COLUMN].shift(lag)

    featured_df["Persentase_Perubahan"] = featured_df[PRICE_COLUMN].pct_change() * 100
    return featured_df


def show_line_chart(chart_df: pd.DataFrame, y_columns: list[str]) -> None:
    if alt:
        long_df = chart_df[[DATE_COLUMN, *y_columns]].melt(
            id_vars=DATE_COLUMN,
            var_name="Jenis Data",
            value_name="Nilai Harga",
        ).dropna(subset=["Nilai Harga"])

        chart = (
            alt.Chart(long_df)
            .mark_line()
            .encode(
                x=alt.X(f"{DATE_COLUMN}:T", title="Tanggal"),
                y=alt.Y("Nilai Harga:Q", title="Harga"),
                color=alt.Color("Jenis Data:N", title="Jenis Data"),
                tooltip=[
                    alt.Tooltip(f"{DATE_COLUMN}:T", title="Tanggal", format="%d-%m-%Y"),
                    alt.Tooltip("Jenis Data:N", title="Jenis Data"),
                    alt.Tooltip("Nilai Harga:Q", title="Harga", format=",.0f"),
                ],
            )
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.line_chart(chart_df.set_index(DATE_COLUMN)[y_columns])


def show_bar_chart(chart_df: pd.DataFrame, x_column: str, y_column: str) -> None:
    if alt:
        chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X(f"{x_column}:N", title=x_column, sort=None),
                y=alt.Y(f"{y_column}:Q", title="Harga"),
                tooltip=[
                    alt.Tooltip(f"{x_column}:N", title=x_column),
                    alt.Tooltip(f"{y_column}:Q", title="Harga", format=",.0f"),
                ],
            )
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.bar_chart(chart_df.set_index(x_column)[y_column])


if not DATA_PATH.exists():
    st.error(f"File data tidak ditemukan: {DATA_PATH}")
    st.stop()


df = add_time_features(load_data(DATA_PATH))

st.title("Exploratory Data Analysis")
st.caption("Analisis data harga Cabai Rawit Merah berdasarkan dataset yang tersedia.")

if COMMODITY_COLUMN in df.columns:
    commodities = sorted(df[COMMODITY_COLUMN].dropna().unique())
    if len(commodities) == 1:
        selected_commodity = commodities[0]
        st.write(f"Komoditas: **{selected_commodity}**")
    else:
        selected_commodity = st.selectbox("Komoditas", commodities, index=0)
    df = df[df[COMMODITY_COLUMN] == selected_commodity].reset_index(drop=True)
else:
    selected_commodity = "Cabai Rawit Merah"

tanggal_awal = df[DATE_COLUMN].min()
tanggal_akhir = df[DATE_COLUMN].max()
harga_awal = df[PRICE_COLUMN].iloc[0]
harga_akhir = df[PRICE_COLUMN].iloc[-1]
selisih_harga = harga_akhir - harga_awal
persen_perubahan = (selisih_harga / harga_awal) * 100

st.subheader("Ringkasan Dataset")

metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)

with metric_col1:
    st.metric("Jumlah Data", len(df))

with metric_col2:
    st.metric("Rata-rata Harga", rupiah(df[PRICE_COLUMN].mean()))

with metric_col3:
    st.metric("Harga Tertinggi", rupiah(df[PRICE_COLUMN].max()))

with metric_col4:
    st.metric("Harga Terendah", rupiah(df[PRICE_COLUMN].min()))

with metric_col5:
    st.metric("Harga Terakhir", rupiah(harga_akhir))

st.info(
    "Dataset berisi data "
    f"**{selected_commodity}** dari **{tanggal_awal:%d-%m-%Y}** "
    f"sampai **{tanggal_akhir:%d-%m-%Y}**."
)

overview_col, quality_col = st.columns([2, 1])

with overview_col:
    display_columns = [
        column
        for column in [DATE_COLUMN, COMMODITY_COLUMN, PRICE_COLUMN]
        if column in df.columns
    ]
    st.dataframe(df[display_columns].head(10), use_container_width=True)

with quality_col:
    missing_count = int(df[display_columns].isna().sum().sum())
    duplicate_count = int(df[display_columns].duplicated().sum())
    st.metric("Jumlah Kolom Asli", len(display_columns))
    st.metric("Missing Value", missing_count)
    st.metric("Duplikat", duplicate_count)

st.divider()

st.subheader("Statistik Deskriptif Harga")

statistics_df = pd.DataFrame(
    {
        "Statistik": [
            "Mean",
            "Median",
            "Minimum",
            "Maximum",
            "Standar Deviasi",
            "Varians",
            "Skewness",
            "Kurtosis",
        ],
        "Nilai": [
            df[PRICE_COLUMN].mean(),
            df[PRICE_COLUMN].median(),
            df[PRICE_COLUMN].min(),
            df[PRICE_COLUMN].max(),
            df[PRICE_COLUMN].std(),
            df[PRICE_COLUMN].var(),
            df[PRICE_COLUMN].skew(),
            df[PRICE_COLUMN].kurt(),
        ],
    }
)

st.dataframe(
    statistics_df.style.format({"Nilai": "{:,.2f}"}),
    use_container_width=True,
)

st.divider()

st.subheader("Tren Harga Harian")
show_line_chart(df[[DATE_COLUMN, PRICE_COLUMN]], [PRICE_COLUMN])

if selisih_harga > 0:
    trend_text = "mengalami kenaikan"
elif selisih_harga < 0:
    trend_text = "mengalami penurunan"
else:
    trend_text = "relatif stabil"

st.success(
    f"Harga dari awal sampai akhir periode {trend_text} sebesar "
    f"**{rupiah(abs(selisih_harga))}** atau **{persen_perubahan:.2f}%**."
)

st.divider()

st.subheader("Moving Average")
ma_df = df[[DATE_COLUMN, PRICE_COLUMN, "MA_7", "MA_30"]].rename(
    columns={
        PRICE_COLUMN: "Harga Aktual",
        "MA_7": "MA 7 Hari",
        "MA_30": "MA 30 Hari",
    }
)
show_line_chart(ma_df, ["Harga Aktual", "MA 7 Hari", "MA 30 Hari"])
st.caption(
    "MA 7 hari membantu membaca perubahan jangka pendek, sedangkan MA 30 hari "
    "memperlihatkan arah tren yang lebih halus."
)

st.divider()

st.subheader("Volatilitas Harga")
volatility_col, change_col = st.columns(2)

with volatility_col:
    volatility_df = df[[DATE_COLUMN, "Std_7"]].rename(
        columns={"Std_7": "Volatilitas 7 Hari"}
    )
    show_line_chart(volatility_df, ["Volatilitas 7 Hari"])
    st.metric("Rata-rata Volatilitas 7 Hari", f"{df['Std_7'].mean():,.2f}")

with change_col:
    change_df = df[[DATE_COLUMN, "Persentase_Perubahan"]].rename(
        columns={"Persentase_Perubahan": "Perubahan Harian (%)"}
    )
    if alt:
        chart = (
            alt.Chart(change_df.dropna())
            .mark_line()
            .encode(
                x=alt.X(f"{DATE_COLUMN}:T", title="Tanggal"),
                y=alt.Y("Perubahan Harian (%):Q", title="Perubahan Harian (%)"),
                tooltip=[
                    alt.Tooltip(f"{DATE_COLUMN}:T", title="Tanggal", format="%d-%m-%Y"),
                    alt.Tooltip("Perubahan Harian (%):Q", title="Perubahan (%)", format=".2f"),
                ],
            )
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.line_chart(change_df.set_index(DATE_COLUMN))
    st.metric("Rata-rata Perubahan Harian", f"{df['Persentase_Perubahan'].mean():.2f}%")

st.divider()

distribution_tab, correlation_tab = st.tabs(
    ["📈 Distribusi Target", "🔗 Korelasi Fitur"]
)

q1 = df[PRICE_COLUMN].quantile(0.25)
q3 = df[PRICE_COLUMN].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outlier_df = df[(df[PRICE_COLUMN] < lower_bound) | (df[PRICE_COLUMN] > upper_bound)]

monthly_df = (
    df.groupby(["Bulan_Angka", "Bulan"], as_index=False)[PRICE_COLUMN]
    .mean()
    .sort_values("Bulan_Angka")
)
highest_month = monthly_df.loc[monthly_df[PRICE_COLUMN].idxmax()]
lowest_month = monthly_df.loc[monthly_df[PRICE_COLUMN].idxmin()]

skew = df[PRICE_COLUMN].skew()
if skew > 0:
    skew_text = "miring ke kanan"
elif skew < 0:
    skew_text = "miring ke kiri"
else:
    skew_text = "simetris"

with distribution_tab:
    target_chart_col, target_info_col = st.columns([2, 1])

    with target_chart_col:
        if alt:
            histogram = (
                alt.Chart(df)
                .mark_bar()
                .encode(
                    x=alt.X(
                        f"{PRICE_COLUMN}:Q",
                        bin=alt.Bin(maxbins=35),
                        title="Harga",
                    ),
                    y=alt.Y("count():Q", title="Jumlah Data"),
                    tooltip=[
                        alt.Tooltip(f"{PRICE_COLUMN}:Q", bin=True, title="Rentang Harga"),
                        alt.Tooltip("count():Q", title="Jumlah Data"),
                    ],
                )
                .properties(height=320)
            )
            st.altair_chart(histogram, use_container_width=True)
        else:
            st.bar_chart(df[PRICE_COLUMN].value_counts(bins=35).sort_index())

    with target_info_col:
        st.metric("Skewness", f"{skew:.2f}")
        st.metric("Batas Bawah Outlier", rupiah(lower_bound))
        st.metric("Batas Atas Outlier", rupiah(upper_bound))
        st.metric("Jumlah Outlier", len(outlier_df))
        st.caption(f"Distribusi harga cenderung {skew_text}.")

    month_col, extreme_col = st.columns(2)

    with month_col:
        st.write("Rata-rata harga per bulan")
        show_bar_chart(monthly_df, "Bulan", PRICE_COLUMN)
        st.caption(
            f"Tertinggi: {highest_month['Bulan']} ({rupiah(highest_month[PRICE_COLUMN])}). "
            f"Terendah: {lowest_month['Bulan']} ({rupiah(lowest_month[PRICE_COLUMN])})."
        )

    with extreme_col:
        st.write("Harga tertinggi dan terendah")
        st.dataframe(
            pd.concat(
                [
                    df.nlargest(5, PRICE_COLUMN)[[DATE_COLUMN, PRICE_COLUMN]].assign(
                        Kategori="Tertinggi"
                    ),
                    df.nsmallest(5, PRICE_COLUMN)[[DATE_COLUMN, PRICE_COLUMN]].assign(
                        Kategori="Terendah"
                    ),
                ],
                ignore_index=True,
            ),
            use_container_width=True,
            height=280,
        )

with correlation_tab:
    correlation_columns = [
        PRICE_COLUMN,
        "Lag_1",
        "Lag_3",
        "Lag_7",
        "Lag_14",
        "Lag_30",
        "MA_7",
        "MA_30",
        "Std_7",
    ]

    correlation_df = df[correlation_columns].corr().round(2)
    correlation_long_df = (
        correlation_df.reset_index(names="Fitur")
        .melt(id_vars="Fitur", var_name="Pembanding", value_name="Korelasi")
    )

    heatmap_col, target_corr_col = st.columns([2, 1])

    with heatmap_col:
        if alt:
            heatmap = (
                alt.Chart(correlation_long_df)
                .mark_rect()
                .encode(
                    x=alt.X("Pembanding:N", title="Fitur Pembanding"),
                    y=alt.Y("Fitur:N", title="Fitur"),
                    color=alt.Color(
                        "Korelasi:Q",
                        scale=alt.Scale(scheme="redblue", domain=[-1, 1]),
                        title="Korelasi",
                    ),
                    tooltip=[
                        alt.Tooltip("Fitur:N", title="Fitur"),
                        alt.Tooltip("Pembanding:N", title="Pembanding"),
                        alt.Tooltip("Korelasi:Q", title="Korelasi", format=".2f"),
                    ],
                )
            )

            labels = (
                alt.Chart(correlation_long_df)
                .mark_text(size=11)
                .encode(
                    x=alt.X("Pembanding:N"),
                    y=alt.Y("Fitur:N"),
                    text=alt.Text("Korelasi:Q", format=".2f"),
                    color=alt.condition(
                        "abs(datum.Korelasi) > 0.55",
                        alt.value("white"),
                        alt.value("black"),
                    ),
                )
            )

            st.altair_chart(
                (heatmap + labels).properties(height=360),
                use_container_width=True,
            )
        else:
            st.dataframe(correlation_df, use_container_width=True)

    target_correlation_df = (
        correlation_df[PRICE_COLUMN]
        .drop(labels=[PRICE_COLUMN])
        .reset_index()
        .rename(columns={"index": "Fitur", PRICE_COLUMN: "Korelasi terhadap Harga"})
    )
    target_correlation_df["Korelasi Absolut"] = target_correlation_df[
        "Korelasi terhadap Harga"
    ].abs()
    target_correlation_df = target_correlation_df.sort_values(
        "Korelasi Absolut",
        ascending=False,
    ).drop(columns="Korelasi Absolut")

    with target_corr_col:
        st.write("Terhadap target Harga")
        st.dataframe(
            target_correlation_df.style.format({"Korelasi terhadap Harga": "{:.2f}"}),
            use_container_width=True,
            height=320,
        )

        strongest_feature = target_correlation_df.iloc[0]
        st.success(
            f"Terkuat: **{strongest_feature['Fitur']}** "
            f"({strongest_feature['Korelasi terhadap Harga']:.2f})."
        )

    st.caption(
        "Nilai mendekati 1 berarti hubungan searah kuat, nilai mendekati -1 "
        "berarti hubungan berlawanan kuat, dan nilai mendekati 0 berarti lemah."
    )
