import numpy as np
from rsi_calculator import calculate_rsi


def generate_trading_signals(data, rsi_period=14, sentiment_threshold=0.1):
    # Calculate RSI
    data['RSI'] = calculate_rsi(data, periods=rsi_period)

    # Calculate Moving Averages
    data['MA_Short'] = data['Close'].rolling(window=7).mean()
    data['MA_Long'] = data['Close'].rolling(window=30).mean()

    # Generate signals based on RSI and Moving Averages
    data['Signal'] = np.where((data['RSI'] < 30) & (data['MA_Short'] > data['MA_Long']), 'Buy',
                              np.where((data['RSI'] > 70) & (data['MA_Short'] < data['MA_Long']), 'Sell', 'Hold'))

    return data
