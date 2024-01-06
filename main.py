import streamlit as st
import json
import pandas as pd
from datetime import datetime, timedelta
from data_fetcher import fetch_data, save_to_json
from data_visualizer import plot_stock_data
from clustering_stock_selection import perform_clustering
from data_processing import preprocess_data

def main():
    # Load the NASDAQ-100 company data from the JSON file
    with open('nasdaq100_companies.json', 'r') as json_file:
        nasdaq_100_data = json.load(json_file)

    # Convert the dictionary keys (company names) into a list for the dropdown
    nasdaq_100_tickers = list(nasdaq_100_data.keys())

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
        print(f"Preprocessed data shape: {preprocessed_data.shape}")
        print(f"Number of tickers: {len(ticker_names)}")
        # Perform clustering and get the result DataFrame
        clustering_result = perform_clustering(preprocessed_data, num_clusters, ticker_names)

        # Display the clustering results
        st.write("Clustering results:")
        for cluster_num in range(num_clusters):
            st.write(f"Cluster {cluster_num + 1}:")
            ticker_in_clustering = clustering_result[clustering_result['Cluster'] == cluster_num]['Ticker'].tolist()
            st.write(ticker_in_clustering)

        # Perform clustering using the last 365 days of data
        # clustering_result = perform_clustering(data, num_clusters, start_date, end_date)

    # Select one stock from each cluster
        selected_stocks = []
        for i in range(num_clusters):
            stocks_in_cluster = clustering_result[clustering_result['Cluster'] == i]['Ticker'].tolist()
            selected_stock = stocks_in_cluster[0]  # Select the first stock in each cluster
            selected_stocks.append(selected_stock)

        # Display the data using the data_visualizer module
        if data is not None:
            st.write(f"Stock Data for {user_input} ({selected_ticker}):")
            st.write(data.head())  # Display the first few rows of the data
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

if __name__ == "__main__":
    main()
