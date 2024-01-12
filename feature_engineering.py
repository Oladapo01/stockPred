"""def create_lag_features(data, n_lag=1):
    for i in range(1, n_lag + 1):
        data['lag_' + str(i)] = data['Close'].shift(i)
    return data

def create_rolling_mean(data, windows_size=5):
    data['Rolling_mean'] = data['Close'].rolling(windows_size).mean()
    return data"""