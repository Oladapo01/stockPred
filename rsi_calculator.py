import numpy as np
import pandas as pd

def calculate_rsi(data, periods=14):
    print("Type of data:", type(data))
    print("Data 'Close' column:", data['Close'])
    # Ensure 'Close' is a Series and contains numeric values
    #if not isinstance(data, pd.DataFrame) or 'Close' not in data.columns:
    #    raise ValueError("'data' must be a DataFrame with a 'Close' column.")
    #if not pd.api.types.is_numeric_dtype(data['Close']):
    #    raise TypeError("'Close' column must contain numeric values.")
    delta = data['Close'].diff()
    gain = (delta.mask(delta < 0, 0)).fillna(0)
    loss = (-delta.mask(delta > 0, 0)).fillna(0)

    avg_gain = gain.rolling(window=periods).mean()
    avg_loss = loss.rolling(window=periods).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi