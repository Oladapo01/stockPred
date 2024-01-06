import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


def plot_temporal_structure(data, ticker):
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Close'], label=f'Close Price for {ticker}')
    plt.title(f'Time series for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    fig = plt.gcf()  # Getting the current figure
    plt.close()  # Close the plot to prevent it from displaying in non-Streamlit environments
    return fig

def plot_distribution(data,ticker):
    plt.figure(figsize=(14, 7))
    sns.distplot(data['Close'], bins=50, kde=True)
    plt.title(f'Distribution of Close Price for {ticker}')
    plt.xlabel('Close Price')
    plt.ylabel('Density')
    fig = plt.gcf()  # Getting the current figure
    plt.close()  # Close the plot to prevent it from displaying in non-Streamlit environments
    return fig

def plot_interval_change(data, ticker, interval='M'):
    data_interval = data['Close'].resample(interval).mean()
    plt.figure(figsize=(14, 7))
    data_interval.pct_change().plot(kind='bar')
    plt.title(f'Change in Distribution Over Intervals for {ticker}')
    plt.xlabel(f'{interval} Interval')
    plt.ylabel('Percentage Change')
    fig = plt.gcf()  # Getting the current figure
    plt.close()  # Close the plot to prevent it from displaying in non-Streamlit environments
    return fig

def plot_candlestick(data, ticker):
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                                            open=data['Open'],
                                            high=data['High'],
                                            low=data['Low'],
                                            close=data['Close'])]
                    )
    fig.update_layout(title=f'Candlestick Chart for {ticker}',
                        yaxis_title='Price',
                        xaxis_title='Date',
                        xaxis_rangeslider_visible=True
                      )
    return fig
