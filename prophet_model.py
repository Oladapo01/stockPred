import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly

def train_prophet_model(data):
    # Ensure 'Date' column is present and in datetime format
    if 'Date' not in data.columns or not pd.api.types.is_datetime64_any_dtype(data['Date']):
        raise ValueError("DataFrame must have a 'Date' column in datetime format.")

    # Rename the columns to 'ds' and 'y'
    df = data.rename(columns={'Date': 'ds', 'Close': 'y'})

    # Initialize and fit the model
    model = Prophet(daily_seasonality=True) # Set to 'True' if the data has daily seasonality
    # Fit the model
    model.fit(df)

    return model

def forecast_with_prophet(model, periods):
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    return forecast

def plot_prophet_forecast(model, forecast):
    # Plot the forecast
    fig = plot_plotly(model, forecast)
    return fig