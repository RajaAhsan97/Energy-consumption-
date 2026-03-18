import os
import joblib
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Import your pipeline
from src.etl.extract import load_data
from src.etl.transform import clean_data
from src.features.build_features import create_datetime_features


def evaluate_model():
    """
    Evaluate trained model on validation data
    """

    # Resolve paths properly
    script_dir = os.path.dirname(os.path.abspath(__file__))

    raw_file = os.path.join(
        script_dir,
        "..", "..",
        "data", "raw",
        "individual_household_electric_power_consumption",
        "household_power_consumption.txt"
    )
    raw_file = os.path.abspath(raw_file)

    print("Loading data from:", raw_file)

    model_path = os.path.join(script_dir, "..", "..", "models", "RandomForest.pkl")
    model_path = os.path.abspath(model_path)


    # =====  ETL =====
    df_raw = load_data(raw_file)
    df_clean = clean_data(df_raw)

    # =====  Feature Engineering =====
    df_features = create_datetime_features(df_clean)

    # =====  Prepare data =====
    X = df_features[['hour', 'day', 'month', 'day_of_week']]
    y = df_features['Global_active_power']

    # same split as training
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =====  Load model =====
    print("Loading model from:", model_path)
    model = joblib.load(model_path)

    # =====  Predict =====
    y_pred = model.predict(X_test)

    # =====  Metrics =====
    rmse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n📊 Evaluation Results:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R2  : {r2:.4f}")

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }


# ===== Run =====
if __name__ == "__main__":
    #script_dir = os.path.dirname(os.path.abspath(__file__))
    evaluate_model()

# run this code 
# python -m src.model.evaluate
