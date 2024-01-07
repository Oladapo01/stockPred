import plotly.graph_objects as go
import streamlit as st
from prophet.plot import plot_plotly


def plot_stock_data(data, ticker):
    # Plots the closing price of a stock using Plotly
    if data is not None:
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name=f'{ticker} Closing Price'))

        fig.update_layout(title=f"{ticker} Closing Price",
                          xaxis_title='Date',
                          yaxis_title='Closing Price',
                          showlegend=True)
        return fig
    else:
        st.error(f"No data to plot for {ticker}")


def plot_prophet_forecast_with_annotations(prophet_model, forecast, periods_to_forecast):
    # Generate a Plotly figure from the Prophet model and forecast
    fig = plot_plotly(prophet_model, forecast)

    # Add actual and predicted labels
    # Adjust the x and y coordinates to place your labels appropriately
    actual_label = go.layout.Annotation(
        x=forecast['ds'].iloc[-periods_to_forecast],
        y=forecast['yhat'].iloc[-periods_to_forecast],
        text="Actual",
        showarrow=True,
        arrowhead=1
    )

    predicted_label = go.layout.Annotation(
        x=forecast['ds'].iloc[-1],
        y=forecast['yhat'].iloc[-1],
        text="Predicted",
        showarrow=True,
        arrowhead=1
    )

    # Add annotations to the figure
    fig.update_layout(annotations=[actual_label, predicted_label])

    return fig