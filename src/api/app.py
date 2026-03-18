# src/api/app.py
import os
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# ===== Load model at startup =====
script_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    script_dir,
    "..", "..",
    "models",
    "RandomForest.pkl"
)
model_path = os.path.abspath(model_path)

print("Loading model from:", model_path)
model = joblib.load(model_path)

# ===== FastAPI setup =====
app = FastAPI(title="Energy Consumption Prediction API")


# ===== Request schema =====
class PredictionRequest(BaseModel):
    hour: int
    day: int
    month: int
    day_of_week: int

# ===== Health check endpoint =====
@app.get("/")
def read_root():
    return {"message": "Energy Consumption Prediction API is running!"}

# ===== API endpoint =====
@app.post("/predict")
def predict(request: PredictionRequest):
    X = np.array([[request.hour, request.day, request.month, request.day_of_week]])
    prediction = model.predict(X)
    return {"predicted_global_active_power": float(prediction[0])}


# run this
# uvicorn src.api.app:app --reload