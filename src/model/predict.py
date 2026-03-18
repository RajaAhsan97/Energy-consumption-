import os
import joblib
import numpy as np


def load_model():
    """
    Load trained model from models folder
    """
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
    return model


def predict(hour: int, day: int, month: int, day_of_week: int):
    """
    Predict Global Active Power based on datetime features
    """
    model = load_model()

    # Create input array
    X = np.array([[hour, day, month, day_of_week]])

    # Predict
    prediction = model.predict(X)

    return float(prediction[0])


if __name__ == "__main__":
    print("Enter input values:")

    hour = 14
    day = 10
    month = 3
    day_of_week = 2

    result = predict(hour, day, month, day_of_week)

    print(f"\n Predicted Global Active Power: {result:.4f}")

# run this code
# python -m src.model.predict