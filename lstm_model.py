from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

def create_lstm_model(data, n_steps, n_features):
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))

    # Create the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=1))
    return model, scaler

def preprocess_lstm_data(data, n_steps):
    # Create the data for LSTM
    X = []
    y = []
    for i in range(n_steps, len(data)):
        X.append(data[i - n_steps:i])
        y.append(data[i])
    X, y = np.array(X), np.array(y)

    # Reshape the data to 3D for LSTM input
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y

def train_lstm_stock_model(stock_data, n_steps=10):
    # Prepare the data
    close_prices = stock_data['Close'].values
    X, y = preprocess_lstm_data(close_prices, n_steps)

    # Create the LSTM model
    model, scaler = create_lstm_model(close_prices, n_steps, 1)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X, y, epochs=20, batch_size=32)
    return model, scaler