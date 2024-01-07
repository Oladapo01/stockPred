from pmdarima import auto_arima
import pandas as pd


def train_arima_mys_tock_model(data, ticker, forecast_date):
    # Ensure the index is a DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        # If the index is not datetime, attempt to convert it
        data.index = pd.to_datetime(data.index)

    forecast_date = pd.to_datetime(forecast_date)
    # ARIMA model expects a uni-variate series
    close_prices = data['Close']

    # Split the data into test and train sets
    train_size = int(len(close_prices) * 0.8)
    train, test = close_prices.iloc[:train_size], close_prices.iloc[train_size:]

    # Fit the ARIMA model
    model = auto_arima(train,
                       start_p=1, start_q=1,
                       max_p=3, max_q=3, m=12,
                       start_P=0,
                       d=None, D=1, trace=True,
                       seasonal=False,
                       error_action='ignore',
                       suppress_warnings=True
                       )
    model.fit(train)

    # Calculate the forecast period
    forecast_period = len(test) + (forecast_date - test.index[-1]).days

    # Generate the forecast index starting from the day after the last date in 'test'
    forecast_index = pd.date_range(start=test.index[-1] + pd.Timedelta(days=1), periods=forecast_period, freq='D')

    # Make predictions on the test set and also for future dates if needed
    forecast_values = model.predict(n_periods=forecast_period)
    print("Forecast values:", forecast_values)
    forecast = pd.Series(forecast_values.values, index=forecast_index, name='Forecast')

    return train, test, forecast