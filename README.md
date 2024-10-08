StockPricesPrediction 2.0

This project predicts the stock prices of selected companies for the next four months using historical data and machine learning models. The application retrieves stock data via APIs, stores the data in CSV files, and applies predictive algorithms to estimate future stock prices.


How it Works

Stock Data Download (stockhistory.py):
This script uses the yfinance library to download historical stock price data.
The stock data is saved into the /data folder as CSV files.
Stock Price Prediction (stockpredict.py):
This script loads the downloaded CSV files, applies a machine learning algorithm (Linear Regression) to predict stock prices for the next 4 months (~120 days).
Predictions are visualized using matplotlib to compare historical and predicted prices.
Main Initialization (__init__.py):
This orchestrates the entire workflow by running stockhistory.py followed by stockpredict.py.
