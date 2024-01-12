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

def trading_signals_visualizer(data):
    # Create a Plotly figure
    fig = go.Figure()

    # Add line for Close prices
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))

    # Add scatter plot for Buy signals
    buys = data[data['Signal'] == 'Buy']
    fig.add_trace(go.Scatter(x=buys.index, y=buys['Close'], mode='markers', name='Buy',
                             marker=dict(color='green', size=10, symbol='triangle-up')))

    # Add scatter plot for Sell signals
    sells = data[data['Signal'] == 'Sell']
    fig.add_trace(go.Scatter(x=sells.index, y=sells['Close'], mode='markers', name='Sell',
                             marker=dict(color='red', size=10, symbol='triangle-down')))

    # Update the layout of the figure
    fig.update_layout(
        title='Trading Signals',
        xaxis_title='Date',
        yaxis_title='Price',
        width=500,  # Set the desired width
        height=400  # Set the desired height
    )

    return fig

def plot_lstm_predictions_with_signals(actual, predicted, signals):
    fig = go.Figure()

    # Add the actual prices trace
    fig.add_trace(go.Scatter(
        x=actual.index,
        y=actual['Actual'],
        mode='lines',
        name='Actual'
    ))
    print(actual.index)

    # Add the predicted prices trace
    fig.add_trace(go.Scatter(
        x=predicted.index,
        y=predicted['Predictions'],
        mode='lines',
        name='Predicted'
    ))
    print(predicted.index)

    # Add markers for buy/sell signals
    for signal in signals.itertuples():
        fig.add_trace(go.Scatter(
            x=[signal.Index],
            y=[signal.Actual if hasattr(signal, 'Actual') else signal.Predictions],
            mode='markers',
            name=f'{signal.Signal} Signal',
            marker=dict(
                color='green' if signal.Signal == 'Buy' else 'red',
                size=10,
                symbol='triangle-up' if signal.Signal == 'Buy' else 'triangle-down'
            )
        ))

    # Customize layout
    fig.update_layout(
        title='LSTM Predictions and Trading Signals',
        xaxis_title='Date',
        yaxis_title='Price'
    )

    return fig


def plot_regression_predictions_with_signals(actual, predicted, signals):
    # Create the figure
    fig = go.Figure()

    # Add the actual prices trace
    fig.add_trace(go.Scatter(x=actual.index, y=actual['Actual'], mode='lines', name='Actual'))

    # Add the predicted prices trace
    fig.add_trace(go.Scatter(x=predicted.index, y=predicted['Predicted'], mode='lines', name='Predicted'))

    # Add markers for buy/sell signals
    fig.add_trace(go.Scatter(x=signals.index, y=signals['Actual'], mode='markers', name='Buy Signal',
                             marker=dict(color='green', size=10, symbol='triangle-up'),
                             customdata=signals['Signal'],
                             hovertemplate='%{y} %{customdata}<extra></extra>'))

    fig.add_trace(go.Scatter(x=signals.index, y=signals['Actual'], mode='markers', name='Sell Signal',
                             marker=dict(color='red', size=10, symbol='triangle-down'),
                             customdata=signals['Signal'],
                             hovertemplate='%{y} %{customdata}<extra></extra>'))

    # Customize layout
    fig.update_layout(title='Regression Predictions and Trading Signals', xaxis_title='Date', yaxis_title='Price')

    return fig


def plot_prophet_forecast_with_signals(forecast, actual, signals):
    fig = go.Figure()

    # Add the actual prices trace
    fig.add_trace(go.Scatter(
        x=actual['Date'],
        y=actual['y'],
        mode='lines',
        name='Actual'
    ))

    # Add the forecasted prices trace
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Forecast'
    ))

    # Add markers for buy/sell signals
    for signal in signals.itertuples():
        fig.add_trace(go.Scatter(
            x=[signal.Date],
            y=[signal.y],
            mode='markers',
            name=f'{signal.Signal} Signal',
            marker=dict(color='green' if signal.Signal == 'Buy' else 'red', size=10, symbol='triangle-up' if signal.Signal == 'Buy' else 'triangle-down')
        ))

    # Customize layout
    fig.update_layout(
        title=f'Prophet Forecast and Trading Signals for {actual.name}',
        xaxis_title='Date',
        yaxis_title='Price'
    )

    return fig

def plot_xgboost_predictions_with_signals(actual, predicted, forecast, signals):
    fig = go.Figure()

    # Add the actual prices trace
    fig.add_trace(go.Scatter(
        x=actual.index,
        y=actual['Actual'],
        mode='lines',
        name='Actual'
    ))

    # Add the predicted prices trace
    fig.add_trace(go.Scatter(
        x=predicted.index,
        y=predicted['Predicted'],
        mode='lines',
        name='Predicted'
    ))

    # Add the forecast trace
    fig.add_trace(go.Scatter(
        x=forecast.index,
        y=forecast['Forecast'],
        mode='lines',
        name='Forecast'
    ))

    # Add markers for buy/sell signals
    for signal in signals.itertuples():
        fig.add_trace(go.Scatter(
            x=[signal.Index],  # Assuming the signals DataFrame has the same index as the actual and predicted
            y=[signal.Actual if signal.Signal != 'Forecast' else signal.Forecast],
            mode='markers',
            name=f'{signal.Signal} Signal',
            marker=dict(
                color='green' if signal.Signal == 'Buy' else 'red' if signal.Signal == 'Sell' else 'blue',
                size=10,
                symbol='triangle-up' if signal.Signal == 'Buy' else 'triangle-down' if signal.Signal == 'Sell' else 'circle'
            )
        ))

    # Customize layout
    fig.update_layout(
        title=f'XGBoost Predictions and Trading Signals for {actual.name}',
        xaxis_title='Date',
        yaxis_title='Price'
    )

    return fig