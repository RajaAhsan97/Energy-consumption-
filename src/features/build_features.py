import pandas as pd
import os

def create_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features from a datetime column.
    """
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    return df