from fastapi.testclient import TestClient
from src.api.app import app   

client = TestClient(app)

def test_predict():
    payload = {
        "hour": 14,
        "day": 10,
        "month": 3,
        "day_of_week": 2
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data