import pandas as pd
import os

def save_data(df: pd.DataFrame, output_path: str) -> str:
    """
    Save processed dataframe to CSV (staged data).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}, shape: {df.shape}")
    return output_path

# Example usage
if __name__ == "__main__":
    from extract import load_data
    from transform import clean_data
    
    raw_df = load_data("../../data/raw/individual+household+electric+power+consumption/household_power_consumption.txt")
    clean_df = clean_data(raw_df)
    save_data(clean_df, "../../data/cleaned/energy_consumption.csv")
