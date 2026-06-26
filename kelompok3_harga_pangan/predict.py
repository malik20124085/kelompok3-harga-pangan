import joblib
import pandas as pd

FEATURES = [
    "Day", "Month", "Year", "DayOfWeek", "Quarter",
    "Lag_1", "Lag_3", "Lag_7", "Lag_14", "Lag_30",
    "MA_7", "MA_30", "Std_7"
]


def load_prediction_assets():
    model = joblib.load("models/model_random_forest.pkl")
    scaler_X = joblib.load("models/scaler_X.pkl")
    scaler_y = joblib.load("models/scaler_y.pkl")

    return model, scaler_X, scaler_y


def load_price_data():
    df = pd.read_excel("data/Data_Harga_Interpolasi.xlsx")
    df["Tanggal"] = pd.to_datetime(df["Tanggal"])
    df = df.sort_values("Tanggal").reset_index(drop=True)

    return df


def add_prediction_features(df):
    df = df.copy()

    df["Tanggal"] = pd.to_datetime(df["Tanggal"])

    df["Day"] = df["Tanggal"].dt.day
    df["Month"] = df["Tanggal"].dt.month
    df["Year"] = df["Tanggal"].dt.year
    df["DayOfWeek"] = df["Tanggal"].dt.dayofweek
    df["Quarter"] = df["Tanggal"].dt.quarter

    # Lag features
    for lag in [1, 3, 7, 14, 30]:
        df[f"Lag_{lag}"] = df["Harga"].shift(lag)

    # Rolling statistics
    df["MA_7"] = df["Harga"].rolling(7).mean()
    df["MA_30"] = df["Harga"].rolling(30).mean()
    df["Std_7"] = df["Harga"].rolling(7).std()

    return df


def predict_next_day():
    model, scaler_X, scaler_y = load_prediction_assets()
    df = add_prediction_features(load_price_data())
    df = df.dropna().reset_index(drop=True)
    last_row = df[FEATURES].iloc[[-1]]

    X_scaled = scaler_X.transform(last_row)
    pred_scaled = model.predict(X_scaled)
    pred = scaler_y.inverse_transform(
        pred_scaled.reshape(-1, 1)
    )

    return float(pred[0][0])


def predict_future_days(days=7):
    model, scaler_X, scaler_y = load_prediction_assets()
    history_df = load_price_data()[["Tanggal", "Harga"]].copy()
    predictions = []

    for _ in range(days):
        last_date = history_df["Tanggal"].iloc[-1]
        next_date = last_date + pd.Timedelta(days=1)
        harga_history = history_df["Harga"]

        feature_row = pd.DataFrame(
            [
                {
                    "Day": next_date.day,
                    "Month": next_date.month,
                    "Year": next_date.year,
                    "DayOfWeek": next_date.dayofweek,
                    "Quarter": next_date.quarter,
                    "Lag_1": harga_history.iloc[-1],
                    "Lag_3": harga_history.iloc[-3],
                    "Lag_7": harga_history.iloc[-7],
                    "Lag_14": harga_history.iloc[-14],
                    "Lag_30": harga_history.iloc[-30],
                    "MA_7": harga_history.tail(7).mean(),
                    "MA_30": harga_history.tail(30).mean(),
                    "Std_7": harga_history.tail(7).std(),
                }
            ]
        )

        X_scaled = scaler_X.transform(feature_row[FEATURES])
        pred_scaled = model.predict(X_scaled)
        predicted_price = scaler_y.inverse_transform(
            pred_scaled.reshape(-1, 1)
        )[0][0]
        predicted_price = round(float(predicted_price))

        predictions.append(
            {
                "Tanggal": next_date,
                "Prediksi Harga": predicted_price,
            }
        )

        history_df = pd.concat(
            [
                history_df,
                pd.DataFrame(
                    [
                        {
                            "Tanggal": next_date,
                            "Harga": predicted_price,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    result_df = pd.DataFrame(predictions)
    result_df["Perubahan"] = result_df["Prediksi Harga"].diff()

    if not result_df.empty:
        last_actual_price = load_price_data()["Harga"].iloc[-1]
        result_df.loc[0, "Perubahan"] = result_df.loc[0, "Prediksi Harga"] - last_actual_price

    result_df["Perubahan (%)"] = result_df["Perubahan"] / (
        result_df["Prediksi Harga"] - result_df["Perubahan"]
    ) * 100
    result_df["Prediksi Harga"] = result_df["Prediksi Harga"].round(0)
    result_df["Perubahan"] = result_df["Perubahan"].round(0)
    result_df["Indikator"] = result_df["Perubahan"].apply(get_trend_indicator)

    return result_df


def get_trend_indicator(change):
    if change > 0:
        return "Naik"
    if change < 0:
        return "Turun"

    return "Stabil"


if __name__ == "__main__":
    hasil = predict_next_day()

    print("Prediksi harga berikutnya:")
    print(f"Rp {hasil:,.2f}")
