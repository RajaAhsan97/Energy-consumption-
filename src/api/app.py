# src/api/app.py
# src/api/app.py
import os
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Energy Consumption Prediction API")

model = None  # ✅ global placeholder

# ===== Build model path =====
script_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    script_dir,
    "..", "..",
    "models",
    "RandomForest.pkl"
)
model_path = os.path.abspath(model_path)


# ✅ Load model at startup (NOT import time)
@app.on_event("startup")
def load_model():
    global model
    print("Loading model from:", model_path)
    model = joblib.load(model_path)
    print("Model loaded successfully!")


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
