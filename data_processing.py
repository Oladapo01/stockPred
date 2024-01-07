import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    data = data.transpose()
    # Scale the data
    scaler = StandardScaler()
    # Handle missing data
    data_filled = data.ffill().bfill()
    normalized_data = scaler.fit_transform(data_filled)
    ticker_names = list(data.index)
    return normalized_data, ticker_names


def process_date_data(data):
    # Check if 'Date' column exists in the DataFrame
    if 'Date' in data.columns:
        # If 'Date' column exists, convert it to datetime and set as index
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
    elif not isinstance(data.index, pd.DatetimeIndex):
        # If there is no 'Date' column and the index is not a DatetimeIndex,
        # raise an error or handle it appropriately
        raise ValueError("DataFrame index is not a DatetimeIndex and no 'Date' column found.")
    return data