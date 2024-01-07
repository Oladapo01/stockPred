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

def rsi_visualizer(data):
    # Visualize the RSI of a stock using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI', yaxis='y2'))
    fig.update_layout(title='RSI',
                      xaxis_title='Date',
                      yaxis_title='RSI',
                      side='right',
                      yaxis=dict(
                          title='RSI',
                          showgrid=False,
                          zeroline=False,
                          showline=False
                    ),
                      showlegend=True)
    return fig

def trading_signals_visualizer(signals_data):
    # Visualize trading signals on the chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=signals_data.index, y=signals_data['Close'], mode='lines', name='Close'))
    fig.add_trace(go.Scatter(x=signals_data.index, y=signals_data['MA_Short'], mode='lines', name='Short MA'))
    fig.add_trace(go.Scatter(x=signals_data.index, y=signals_data['MA_Long'], mode='lines', name='Long MA'))

    # Add markers for Buy/Sell signals
    buy_signals = signals_data[signals_data['Signal'] == 'Buy']
    sell_signals = signals_data[signals_data['Signal'] == 'Sell']
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers', name='Buy Signal', marker=dict(color='green')))
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers', name='Sell Signal', marker=dict(color='red')))

    # Update layout if needed, e.g., title, x-axis label, etc.
    fig.update_layout(title='Trading Signals', xaxis_title='Date', yaxis_title='Price')

    return fig