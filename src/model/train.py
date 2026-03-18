# Option A: Train all models and pick the best (automated selection)

# Pros: Automated model selection, experiment tracking.

# Cons: Longer runtime, heavier computation.

# train.py flow:
# features_df → train LogisticRegression
#              → train RandomForest
#              → train GradientBoosting
# compare metrics → save best model

# Option B: Train only the best model from notebook

# Pros: Faster, simpler pipeline.

# Cons: If data distribution changes, the model choice is fixed unless manually updated.

# train.py flow:
# features_df → train RandomForest (or best model) → save model

import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Import ETL + feature functions
from src.etl.extract import load_data
from src.etl.transform import clean_data
from src.features.build_features import create_datetime_features
from src.etl.load import save_data

mlflow.set_tracking_uri("file:///E:/Saylani AI and data science course/codes/MLOps/Energy consumption_MLOps/mlruns")
mlflow.set_experiment("Energy Consumption")

# ===== Helper functions =====
def train_models(X_train, y_train, X_val, y_val):
    """
    Train multiple regression models and return the best one (based on RMSE)
    """
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    best_model = None
    best_rmse = float("inf")
    model_scores = {}


    for name, model in models.items():
        with mlflow.start_run(run_name=name) as run:
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            rmse = mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)

            # Log everything
            mlflow.log_param("model_name", name)
            mlflow.log_metrics({
                "rmse": rmse,
                "mae": mae,
                "r2": r2
            })

            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            model_scores[name] = {"RMSE": rmse, "MAE": mae, "R2": r2}
            print(f"{name} -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                best_model_name = name
                best_run_id = run.info.run_id

    print(f"\nBest model: {best_model_name} with RMSE: {best_rmse:.4f}")

    # REGISTER BEST MODEL HERE
    model_uri = f"runs:/{best_run_id}/model"

    mlflow.register_model(
        model_uri=model_uri,
        name="EnergyForecastModel"
    )

    return best_model, best_model_name, model_scores

def save_model(model, model_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    output_dir = os.path.join(script_dir, "..", "..", "models")
    output_dir = os.path.abspath(output_dir)

    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, f"{model_name}.pkl")
    joblib.dump(model, model_path)
    print(f"Saved best model to {model_path}")
    return model_path

# ===== Main Pipeline =====
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Extract raw data

    raw_file = os.path.join(
        script_dir,       # src/model/
        "..",             # src/
        "..",             # project root
        "data", "raw", 
        "individual_household_electric_power_consumption",
        "household_power_consumption.txt"
    )
    raw_file = os.path.abspath(raw_file)
    
    # Extract
    df_raw = load_data(raw_file)

    # Transform / clean
    df_clean = clean_data(df_raw)

    # Feature engineering
    df_features = create_datetime_features(df_clean)

    # Optional: save features
    features_output = os.path.join(script_dir, "..", "..", "data", "cleaned", "energy_consumption.csv")
    features_output = os.path.abspath(features_output)
    save_data(df_features, features_output)

    # Split data
    X = df_features[['hour','day','month','day_of_week']]
    y = df_features['Global_active_power']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train all models and pick the best
    best_model, best_model_name, all_scores = train_models(X_train, y_train, X_val, y_val)

    # Save the best model
    save_model(best_model, best_model_name)

# run this code using
# python -m src.model.train 

# install mlflow
#pip install mlflow -i https://pypi.tuna.tsinghua.edu.cn/simple

# after training run
# mlflow ui --backend-store-uri "file:///E:/Saylani AI and data science course/codes/MLOps/Energy consumption_MLOps/mlruns"
