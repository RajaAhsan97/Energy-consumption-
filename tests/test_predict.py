from fastapi.testclient import TestClient
from src.api.app import app

def test_predict():
    with TestClient(app) as client:   #  ensures startup event runs
        payload = {
            "hour": 14,
            "day": 10,
            "month": 3,
            "day_of_week": 2
        }

        response = client.post("/predict", json=payload)

        print("Response JSON:", response.json())

        assert response.status_code == 200
        data = response.json()
        assert "predicted_global_active_power" in data
