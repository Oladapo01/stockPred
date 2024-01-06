import pandas as pd
import yfinance as yf
import json


def fetch_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        return data
    except Exception as e:
        print(f"Error fetching data for stock: {ticker}")
        print(e)
        return pd.DataFrame()


def save_to_json(data, group):
    # Ensuring the 'Ticker' is present in the data
    if 'Ticker' not in data.columns:
        data['Ticker'] = data.index

    # Use DateTime index as 'Date' column
    data['Date'] = data.index

    # Convert the data to a dictionary
    data_dict = data.to_dict(orient='split')


    # Save the data to a JSON file with a specific naming format
    file_name = f"Group_{group}_all_stock_data.json"
    with open(file_name, 'w') as json_file:
        json.dump(data_dict, json_file, indent=4)
    print(f"Data for Group {group} appended to {file_name}")
