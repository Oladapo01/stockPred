import plotly.graph_objects as go
import streamlit as st

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
