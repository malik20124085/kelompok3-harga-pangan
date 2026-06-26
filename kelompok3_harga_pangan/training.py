import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from sklearn.ensemble import RandomForestRegressor

from preprocessing import (
    load_data,
    create_features,
    create_lag_features,
    scale_data
)


def train_random_forest():

    print("Memuat dataset...")

    df = load_data(
        "data/Data_Harga_Interpolasi.xlsx"
    )

    df = create_features(df)

    df = create_lag_features(df)

    X_scaled, y_scaled, scaler_X, scaler_y = scale_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y_scaled,
        test_size=0.2,
        random_state=42
    )

    print("Training Random Forest...")

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = mean_squared_error(
        y_test,
        y_pred
    ) ** 0.5

    print("\n===== HASIL EVALUASI =====")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")

    print("\nMenyimpan model...")

    joblib.dump(
        model,
        "models/model_random_forest.pkl"
    )

    joblib.dump(
        scaler_X,
        "models/scaler_X.pkl"
    )

    joblib.dump(
        scaler_y,
        "models/scaler_y.pkl"
    )

    print("Selesai.")

    return model


if __name__ == "__main__":
    train_random_forest()