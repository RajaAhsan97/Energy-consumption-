import pandas as pd
import os

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load raw data from CSV or other source.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist")
    
    df = pd.read_csv(file_path, sep=';')
    print(f"Data loaded from {file_path}, shape: {df.shape}")
    return df

if __name__ == "__main__":
    df = load_data("../../data/raw/individual+household+electric+power+consumption/household_power_consumption.txt")
    print(df.head())