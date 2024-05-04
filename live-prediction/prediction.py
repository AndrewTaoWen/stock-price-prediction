import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib

# Set current directory and list files
cwd = os.getcwd()
files = os.listdir()

# Load data
file_path = os.path.join(cwd, 'data', 'stock_data.json')
with open(file_path, 'r') as file:
    data = json.load(file)

# Data processing
dates = [date for date, values in data["Time Series (1min)"].items()]
prices = [float(values["4. close"]) for date, values in data["Time Series (1min)"].items()]
volumes = [int(values["5. volume"]) for date, values in data["Time Series (1min)"].items()]

df = pd.DataFrame({"Date": pd.to_datetime(dates), "Close": prices, "Volume": volumes})
df.set_index('Date', inplace=True)

# Prepare data for model
X = df[['Close', 'Volume']]
y = df['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model handling
try:
    model = joblib.load('linear_regression_model.pkl')
except FileNotFoundError:
    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, 'linear_regression_model.pkl')

# Predictions and evaluation
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5

# Set the date formatter for all plots
date_formatter = mdates.DateFormatter('%Y-%m-%d')

# Plot 1: Actual Stock Prices
plt.figure(figsize=(20, 12))
plt.plot(df.index, df['Close'], color='blue', label='Actual')
plt.title('Actual Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.gca().xaxis.set_major_formatter(date_formatter)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Plot 2: Predicted Stock Prices
plt.figure(figsize=(20, 12))
plt.scatter(X_test.index, predictions, color='red', label='Predicted')
plt.title('Predicted Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.gca().xaxis.set_major_formatter(date_formatter)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Plot 3: Actual vs Predicted Stock Prices
plt.figure(figsize=(20, 12))
plt.scatter(X_test.index, y_test, color='blue', label='Actual')
plt.scatter(X_test.index, predictions, color='red', label='Predicted')
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.gca().xaxis.set_major_formatter(date_formatter)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()