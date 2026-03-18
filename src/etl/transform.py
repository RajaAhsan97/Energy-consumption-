import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess raw dataframe.
    - Replace missing values
    - Convert columns to proper types
    """
    # Replace "?" with NaN
    df.replace("?", np.nan, inplace=True)
    
    # Convert numeric columns
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col])
            except ValueError:
                pass  # Non-numeric column remains as object
    
    print(df.columns)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])


    # Fill missing numeric values with median
    for col in df.select_dtypes(include=np.number).columns:
        df[col].fillna(df[col].median(), inplace=True)
    
    print(f"Data cleaned. Missing values remaining: {df.isna().sum().sum()}")
    return df

# Example usage
if __name__ == "__main__":
    from extract import load_data
    raw_df = load_data("../../data/raw/individual+household+electric+power+consumption/household_power_consumption.txt")
    clean_df = clean_data(raw_df)
    print(clean_df.head())