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