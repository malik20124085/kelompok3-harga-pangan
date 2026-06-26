import pandas as pd
from sklearn.preprocessing import MinMaxScaler


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
    "Std_7"
]


def load_data(filepath):
    """
    Membaca dataset Excel
    """

    df = pd.read_excel(filepath)

    df["Tanggal"] = pd.to_datetime(df["Tanggal"])

    return df


def create_features(df):
    """
    Membuat fitur tanggal
    """

    df["Day"] = df["Tanggal"].dt.day
    df["Month"] = df["Tanggal"].dt.month
    df["Year"] = df["Tanggal"].dt.year
    df["DayOfWeek"] = df["Tanggal"].dt.dayofweek
    df["Quarter"] = df["Tanggal"].dt.quarter

    return df


def create_lag_features(df):
    """
    Membuat lag features dan rolling statistics
    """

    for lag in [1, 3, 7, 14, 30]:
        df[f"Lag_{lag}"] = df["Harga"].shift(lag)

    df["MA_7"] = df["Harga"].rolling(window=7).mean()
    df["MA_30"] = df["Harga"].rolling(window=30).mean()
    df["Std_7"] = df["Harga"].rolling(window=7).std()

    df = df.dropna().reset_index(drop=True)

    return df


def scale_data(df):
    """
    Scaling fitur dan target
    """

    X = df[FEATURES]

    y = df[["Harga"]]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y).ravel()

    return (
        X_scaled,
        y_scaled,
        scaler_X,
        scaler_y
    )