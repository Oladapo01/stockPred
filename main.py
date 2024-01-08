import numpy as np
import streamlit as st
import warnings
import json
import pandas as pd
import altair as alt
from datetime import datetime, timedelta
from data_fetcher import fetch_data, save_to_json
from data_visualizer import plot_stock_data, plot_prophet_forecast_with_annotations, trading_signals_visualizer
from clustering_stock_selection import perform_clustering
from data_processing import preprocess_data, process_date_data
from eda import plot_temporal_structure, plot_distribution, plot_interval_change, plot_candlestick
from arima_model import train_arima_mys_tock_model
from regressor_model import train_regressor_model,predict, evaluate
from prophet_model import train_prophet_model, forecast_with_prophet
from sentiment_analysis import fetch_news, analyze_news_sentiment
from rsi_calculator import calculate_rsi
from trading_signals import generate_trading_signals
from xgboost_model import train_xgboost_model, forecast_future_months
from stocks_corr import calculate_correlation_matrix, top_correlated_stocks
from lstm_model import train_lstm_stock_model, preprocess_lstm_data


# Initialize session state variables
if 'order_status' not in st.session_state:
    st.session_state.order_status = ''

# Define the callback functions for the buttons
def place_buy_order():
    st.session_state.order_status = 'buy'

def place_sell_order():
    st.session_state.order_status = 'sell'

def main():
    st.title("Solent Financial Technology (SOLFINTECH)")
    # Load the NASDAQ-100 company data from the JSON file
    with open('nasdaq100_companies.json', 'r') as json_file:
        nasdaq_100_data = json.load(json_file)
    # Suppress specific FutureWarnings from Prophet and Plotly
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
    # Main page tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(
        ["Stock Overview", "Clustering", "Correlation", "EDA", "ARIMA", "LSTM", "Regressor", "Prophet", "XGBoost",
         "Sentiment & Signals"])

    # Convert the dictionary keys (company names) into a list for the dropdown
    nasdaq_100_tickers = list(nasdaq_100_data.keys())
    with tab1:
        # Streamlit user input for company name using dropdown
        user_input = st.selectbox('Select a company', nasdaq_100_tickers)

        # Streamlit user input for start and end dates
        start_date = st.date_input("Select start date", pd.to_datetime('2019-01-01'))
        end_date = st.date_input("Select end date", datetime.today(), max_value=datetime.today())

        # Fetch the ticker symbol corresponding to the selected company
        selected_ticker = nasdaq_100_data[user_input]

        # Fetch the data using the data_fetcher module
        data = fetch_data(selected_ticker, start_date, end_date)
        # Create a Plotly figure
        fig = plot_stock_data(data, user_input)  # This function should return a Plotly figure

        # Display the Plotly figure
        st.plotly_chart(fig)
    with tab2:
        # Fetch the data for all stocks
        all_data = pd.DataFrame()
        ticker_names = []
        start_date = datetime.today() - timedelta(days=365)
        end_date = datetime.today()
        num_clusters = 4

        for ticker in nasdaq_100_data.values():
            try:
                stock_data = fetch_data(ticker, start_date, end_date)
                if not stock_data.empty:
                    # Use the 'Close' price for each day
                    all_data[ticker] = stock_data['Close']
            except Exception as e:
                print(f"Error fetching data for stock: {ticker}: {e}")
                continue

        # Check if all_data is empty or not
        if all_data.empty:
            st.error("No data available to process.")
        else:
            st.success("Data successfully fetched.")
            # Preprocess the data before clustering
            preprocessed_data, ticker_names = preprocess_data(all_data)
            # print(f"Preprocessed data shape: {preprocessed_data.shape}")
            # print(f"Number of tickers: {len(ticker_names)}")
            # Perform clustering and get the result DataFrame
            clustering_result = perform_clustering(preprocessed_data, num_clusters, ticker_names)

            # Display the clustering results
            st.write("Clustering results:")
            for cluster_num in range(num_clusters):
                st.write(f"Cluster {cluster_num + 1}:")
                ticker_in_clustering = clustering_result[clustering_result['Cluster'] == cluster_num]['Ticker'].tolist()
                st.write(ticker_in_clustering)

            # Select one stock from each cluster
            selected_stocks = []
            # Fetch the data for the selected stock
            selected_data = fetch_data(selected_ticker, start_date, end_date)
            for i in range(num_clusters):
                stocks_in_cluster = clustering_result[clustering_result['Cluster'] == i]['Ticker'].tolist()
                selected_stock = stocks_in_cluster[0]  # Select the first stock in each cluster
                selected_stocks.append(selected_stock)

            # Display the data using the data_visualizer module
            if data is not None:
                st.write(f"Stock Data for {user_input} ({selected_ticker}):")
                st.write(selected_data.head())  # Display the first few rows of the data
                plot_stock_data(data, user_input)  # Plot the closing price

                # Perform clustering and stock selection
                selected_stocks = selected_stocks = perform_clustering(preprocessed_data, num_clusters, ticker_names)

                # Display selected stocks for each group
                st.write("Selected stocks for each group:")
                st.write(selected_stocks)

                # Save the data to CSV file
                if st.button("Save Data to JSON"):
                    # Assuming you have a 'group' variable representing the group number (1 to 4)
                    group = 1  # Replace with the actual group number
                    save_to_json(data, group)  # Removed the selected_ticker argument
            else:
                st.error(f"No data available for {user_input} ({selected_ticker}).")


            selected_stocks = clustering_result.groupby('Cluster')['Ticker'].first().tolist()

    with tab3:
        # Calculate correlation matrix
        corr_matrix = calculate_correlation_matrix(all_data)
        correlated_stocks = top_correlated_stocks(corr_matrix, selected_stocks)
        # Ensure selected_stocks are in the index of all_data before computing the correlation
        assert all(stock in all_data.columns for stock in selected_stocks), "Selected stocks must be in all_data columns"

        # Present the top-10 correlated stocks for each stock
        for stock, (top_positive, top_negative) in correlated_stocks.items():
            st.write(f"Stock: {stock}")
            st.write(f"Top positive correlation: {top_positive}")
            st.write(f"Top negative correlation: {top_negative}")

    with tab4:
        # EDA
        # Perform EDA for each selected stocks
        for stock in selected_stocks:
            stock_data = fetch_data(stock, start_date, end_date)
            st.write(f"EDA for {stock}:")
            # Temporal structure
            temporal = plot_temporal_structure(stock_data, stock)
            st.plotly_chart(temporal, use_container_width=True)
            # Distribution of observations
            distribution = plot_distribution(stock_data, stock)
            st.plotly_chart(distribution, use_container_width=True)
            # Interval change
            interval_change = plot_interval_change(stock_data, stock, interval='M')
            st.plotly_chart(interval_change, use_container_width=True)
            # Candlestick chart
            candlestick_fig = plot_candlestick(stock_data, stock)
            st.plotly_chart(candlestick_fig, use_container_width=True)

    with tab5:
        # ARIMA model
        # After selecting a stock, perform ARIMA model
        for i, stock in enumerate(selected_stocks):
            # Define the minimum and maximum dates for the forecast date input
            min_date = datetime.today() - timedelta(days=365)  # 1 year ago from today
            max_date = datetime.today() + timedelta(days=365)  # 1 year into the future


            # Unique key for each date_input using the stocks' ticker symbol
            forecast_date_key = f"Forecast_date_{stock}_{i}"
            # Now you can use min_date and max_date in your date_input widget
            forecast_date = st.date_input("Select date for forecast",
                                          min_value=min_date,
                                          max_value=max_date,
                                          key=forecast_date_key)

            stock_data = fetch_data(stock, start_date, end_date)
            train, test, forecast = train_arima_mys_tock_model(stock_data, stock, forecast_date)
            # Check if the first forecast date is right after the last date in 'test'
            if forecast.index[0] != test.index[-1] + pd.Timedelta(days=1):
                # If there's a gap, fill in the gap with NaNs before the forecast
                gap_index = pd.date_range(start=test.index[-1] + pd.Timedelta(days=1),
                                          end=forecast.index[0] - pd.Timedelta(days=1),
                                          freq='D')
                gap_series = pd.Series([None] * len(gap_index), index=gap_index)
                forecast = pd.concat([gap_series, forecast])

            # Combine the actual and forecast data into a single DataFrame
            combined_data = pd.concat([test.to_frame(name='Actual'), forecast.to_frame(name='Forecast')], axis=1)


            # Ensure the forecast starts the day after the last actual data point
            if forecast.index[0] == test.index[-1]:
                forecast = forecast[1:]

            # Combine the actual and forecast data
            combined_data = test.copy()  # make a copy of the actual data

            # Add forecast data as a new column, this will create NaN for the 'test' period
            combined_data = combined_data.to_frame(name='Actual')
            forecast = forecast.to_frame(name='Forecast')

            # Now combine using the date index; this will align the forecast data next to the actual data
            combined_data = combined_data.join(forecast, how='outer')
            # Convert the DataFrame to a long-form or 'tidy' format
            combined_data_long = combined_data.reset_index().melt('index', var_name='type', value_name='price')

            # Create an Altair chart object
            chart = alt.Chart(combined_data_long).mark_line().encode(
                x='index:T',  # We're assuming 'index' is a datetime column for the x-axis
                y=alt.Y('price:Q', scale=alt.Scale(zero=False)),  # The 'price' column for the y-axis
                color='type:N'  # Differentiate lines by 'type' which is 'Actual' or 'Forecast'
            ).properties(
                width=300,  # Set the width of the chart
                height=250,  # Set the height of the chart
            )

            # Display the chart in Streamlit
            st.write(f"ARIMA model for {stock}:")
            st.altair_chart(chart, use_container_width=True)


    with tab6:
        # LSTM model
        # After selecting a stock, perform LSTM model
        for stocks in selected_stocks:
            n_steps = 10
            # Fetching the data
            stock_data = fetch_data(stocks, start_date, end_date)

            # Training the LSTM model
            model, scaler = train_lstm_stock_model(stock_data, n_steps=n_steps)

            # Prepare the data for plotting
            close_prices = stock_data['Close'].values
            scaler_data = scaler.transform(close_prices.reshape(-1, 1))
            X, _ = preprocess_lstm_data(scaler_data.ravel(), n_steps)

            # Predicting with the LSTM model
            predictions = model.predict(X)
            # Inverse transform the predictions
            predictions = scaler.inverse_transform(predictions)

            # Prepare the actual and predicted for plotting
            actual = pd.DataFrame(close_prices[n_steps:], columns=['Actual'])
            predicted = pd.DataFrame(predictions.ravel(), columns=['Predictions'], index=actual.index)

            # Combining and plot the actual and predicted data
            combined_data = pd.concat([actual, predicted], axis=1)
            st.write(f"LSTM model for {stocks}:")
            st.line_chart(combined_data)


    with tab7:
        # Regressor model
        # After selecting a stock, perform regressor model
        for stock in selected_stocks:
            # Define date range for the last year
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            # Fetch the data
            stock_data = fetch_data(stock, start_date, end_date)

            # Preprocess the data
            stock_data = process_date_data(stock_data)

            # Split the data into train and test sets
            # Assuming the 'preprocess_data' function sets the 'Date' column as the index
            split_date = start_date + timedelta(days=int(len(stock_data) * 0.8))
            train, test = stock_data[:split_date], stock_data[split_date:]

            # Create the features and target DataFrames
            X_train, y_train = train.drop(columns=['Close']), train['Close']
            X_test, y_test = test.drop(columns=['Close']), test['Close']

            # Train the model
            model = train_regressor_model(X_train, y_train)

            # Predict the prices
            y_pred = predict(model, X_test)

            # Evaluate the model
            rmse = evaluate(y_test, y_pred)
            st.write(f"Regressor model for {stock}:")
            st.write(f"RMSE: {rmse}")

            # Plot the actual and predicted prices
            combined_data = pd.concat([y_test.to_frame(name='Actual'), pd.Series(y_pred, index=y_test.index, name='Predicted')], axis=1)
            st.line_chart(combined_data)


    with tab8:
        # Prophet model
        # After selecting a stock, performing prophet model
        for stock in selected_stocks:
            stock_data = fetch_data(stock, start_date, end_date)

            # Ensuring the 'Date' column exists and convert it to datetime format if necessary
            if 'Date' not in stock_data.columns:
                stock_data['Date'] = pd.to_datetime(stock_data.index)

            # Train the model
            prophet_model = train_prophet_model(stock_data)

            # Define the number of days to forecast
            periods_to_forecast = 365
            # Generate a forecast
            forecast = forecast_with_prophet(prophet_model, periods=periods_to_forecast)

            # Plot the forecast
            # Plot the forecast with annotations
            fig = plot_prophet_forecast_with_annotations(prophet_model, forecast, periods_to_forecast)

            # Display the plots
            st.write(f"Prophet model for {stock}:")

            st.plotly_chart(fig, use_container_width=True)

    with tab9:
        # XGBoost model
        # After selecting a stock, perform XGBoost model
        for stock in selected_stocks:
            # Fetch the data
            stock_data = fetch_data(stock, start_date, end_date)

            # Preprocess the data
            stock_data = process_date_data(stock_data)

            # Train the model
            model, rmse, X_test, y_test, y_pred = train_xgboost_model(stock_data, 'Close')

            # Evaluate the model
            st.write(f"XGBoost model for {stock}:")

            # Get the most recent price from your data
            # Plot the actual and predicted prices
            combined_data = pd.concat([y_test.to_frame(name='Actual'), pd.Series(y_pred, index=y_test.index, name='Predicted')], axis=1)

            # Forecast future months
            months = 1
            last_known_data = X_test.tail(1)  # Get the last known data as a DataFrame

            # Creating a new DataFrame for the forecast with the date as the index
            next_day_date = pd.to_datetime('today').normalize() + pd.Timedelta(days=1)  # Normalize to remove the time component

            # Forecast for the next day
            next_day_forecast = forecast_future_months(model, X_test.tail(1), 1)  # Create a DataFrame for the forecast with the date as the index
            forecast_df = pd.DataFrame({'Forecast': next_day_forecast}, index=[next_day_date])
            # Appending the forecast to the combined_data DataFrame
            # Make sure to only append it if it doesn't already exist
            if next_day_date not in combined_data.index:
                combined_data.loc[next_day_date, 'Forecast'] = next_day_forecast[0]

            future_predictions = forecast_future_months(model, last_known_data, 1)

            future_dates = pd.date_range(start=last_known_data.index[-1], periods=months, freq='D')
            print(future_dates)
            future_df = pd.DataFrame(future_predictions, index=future_dates, columns=['Forecast'])
            print(future_df)

            # Combine the historical and forecast data
            # Concatenate the forecast DataFrame to the combined_data DataFrame
            combined_data = pd.concat([combined_data, forecast_df])

            # Sort the index after appending to avoid plotting issues
            combined_data.sort_index(inplace=True)
            # Display the forecast for the next day
            st.markdown(f"Forecast for {next_day_date.strftime('%Y-%m-%d')}: ${next_day_forecast[0]:.2f}")
            # Display current price in bold if available
            current_price = stock_data['Close'].iloc[-1]
            if pd.notna(current_price):
                st.markdown(f"**Current Price: ${current_price:.2f}**")
            # Plot the data
            st.line_chart(combined_data[['Actual', 'Predicted', 'Forecast']])

    with tab10:
        # Sentiment analysis
        # Fetch news data from Alpha Vantage
        for stock in selected_stocks:
            api_key = '90P4SKTLT5H2GUJ9'
            # Ensuring 'Date' column exists and convert it to datetime format if necessary
            if 'Date' not in stock_data.columns:
                stock_data['Date'] = pd.to_datetime(stock_data.index)
            # Print the columns to verify 'Close' is present
            print(f"Columns before RSI calculation: {stock_data.columns}")
            # Calculate RSI
            rsi_values = calculate_rsi(stock_data)
            stock_data['RSI'] = rsi_values
            # Print the columns again to verify 'Close' is still present
            print(f"Columns before generating signals: {stock_data.columns}")

            news_data = fetch_news(api_key, stock)
            # Analyze the sentiment of the news
            sentiments = analyze_news_sentiment(news_data)
            average_sentiment = np.mean(sentiments)
            # Generate trading signals
            # After generating trading signals
            signals_data = generate_trading_signals(stock_data)
            fig = trading_signals_visualizer(signals_data)

            # Displaying the plot with Streamlit
            st.write(f"Trading signals for {stock}:")
            st.plotly_chart(fig)
            # Plot the sentiment of the news
            st.write(f"Sentiment analysis for {stock}:")


            # Plot and display RSI and sentiment analysis results in Streamlit
            st.line_chart(stock_data[['Close', 'RSI']])

            # After selecting a stock, performing analysis, generating signals, etc.
        for stock in selected_stocks:
            # Assuming `signals_data` is a DataFrame with 'Signal' column
            # representing the Buy/Sell/Hold signals
            signals_data = generate_trading_signals(stock_data)

            # Display the latest signal and advice
            latest_signal = signals_data['Signal'].iloc[-1]
            if latest_signal == 'Buy':
                st.success(f"🟢 Buy Advice: Consider buying {stock}.")
            elif latest_signal == 'Sell':
                st.error(f"🔴 Sell Advice: Consider selling {stock}.")
            else:
                st.info(f"🔵 Hold Advice: Maintain your position in {stock}.")

            # Display trading signals
            st.write(f"Trading signals for {stock}:")
            st.dataframe(signals_data[['Close', 'RSI', 'MA_Short', 'MA_Long', 'Signal']])

            # Request user input for order quantity and place an order
            order_qty = st.number_input(f"Enter quantity for {stock}:", min_value=0, value=0, step=1)

            # Buttons for placing orders
            st.button(f"Place Buy order for {stock}", on_click=place_buy_order)
            st.button(f"Place Sell order for {stock}", on_click=place_sell_order)

if __name__ == "__main__":
    main()
