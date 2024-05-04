from flask import Flask, jsonify, request
import requests
import os

# app = Flask(__name__)

# Your Alpha Vantage API key
# API_KEY = os.environ["API_KEY"]
API_KEY = "0DDXA37ZRYDYH4RP"

# Function to get live stock price
def get_live_stock_price(symbol):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&outputsize=full&symbol={symbol}&interval=1min&apikey={API_KEY}'
    response = requests.get(url)
    return response.json()

import json
import pandas as pd

# Function to save data to a JSON file
def save_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

# Function to save data to an Excel file
def save_to_excel(data, filename):
    df = pd.DataFrame.from_dict(data)
    df.to_excel(filename, index=False)

# Define a route to get live stock price
def stock_price(symbol):
    if symbol:
        price_data = get_live_stock_price(symbol)
        print(price_data, len(price_data))
        save_to_json(price_data, 'stock_data.json')  # Save to JSON file
        save_to_excel(price_data, 'stock_data.xlsx')  # Save to Excel file
    
s = input("Enter the stock symbol: ")

stock_price(s)
