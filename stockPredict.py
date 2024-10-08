import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import os
from datetime import timedelta


class stockPredict():
    #initiated base variables and check the csv data files
    def __init__(self):
        self.data_dir = 'data'
        self.model_dir = 'models'
        os.makedirs(self.model_dir, exist_ok=True)
    
        # Process each CSV file in the data directory
        for stock_file in os.listdir(self.data_dir):
            if stock_file.endswith('.csv'):
                self.process_stock(stock_file)

    # Creating datasets and return numpy matrix
    def create_dataset(self, data, time_step=60):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    # Function to process, train and show individual stock
    def process_stock(self, stock_filename):
        model_filename = os.path.join(self.model_dir, f"{os.path.splitext(stock_filename)[0]}_lstm_model.h5")

        # Collecting csv files
        print(f"Loading data for {stock_filename}...")
        df = pd.read_csv(os.path.join(self.data_dir, stock_filename), parse_dates=['Date'], index_col='Date')
        
        #Data Preprocessing
        data = df['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        print("Creating datasets...")
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]

        # Training and testing datasets
        time_step = 60
        X_train, y_train = self.create_dataset(train_data, time_step)
        X_test, y_test = self.create_dataset(test_data, time_step)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Check if the model exists
        if os.path.exists(model_filename):
            print(f"Loading the saved model from {model_filename}...")
            model = load_model(model_filename)
        else:
            # Step 3: Model Design
            print("Creating a new model...")
            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
            model.add(LSTM(units=50))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='adam')

            # Step 4: Training
            print("Starting training")
            model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=1)

            # Save the trained model
            print(f"Saving the model to {model_filename}...")
            model.save(model_filename)

        # Step 5: Prediction
        print("Predicting stock prices")
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)
        y_train = scaler.inverse_transform([y_train])
        y_test = scaler.inverse_transform([y_test])

        # Predict the next 60 days (approximately 2 months)
        last_60_days = scaled_data[-time_step:]
        future_input = last_60_days.reshape(1, time_step, 1)

        # List to store future predictions
        future_predictions = []

        # Predict for the next 60 days
        for _ in range(60):
            future_pred = model.predict(future_input)
            future_predictions.append(future_pred[0, 0])
            future_input = np.append(future_input[:, 1:, :], np.reshape(future_pred, (1, 1, 1)), axis=1)

        future_predictions = np.array(future_predictions).reshape(-1, 1)
        future_predictions = scaler.inverse_transform(future_predictions)

        # Generate future dates
        last_date = df.index[-1]
        future_dates = pd.date_range(last_date + timedelta(days=1), periods=60, freq='B')

        # Step 6: Plotting the Results
        print("Showing results")
        plt.figure(figsize=(14, 6))
        plt.plot(df.index, data, label='Actual Stock Price', color='blue')
        plt.plot(df.index[time_step:time_step + len(train_predict)], train_predict, label='Train Prediction', color='green')
        
        test_start_index = len(train_data) + time_step
        plt.plot(df.index[test_start_index:test_start_index + len(test_predict)], test_predict, label='Test Prediction', color='red')

        plt.plot(future_dates, future_predictions, label='Future Predictions', color='orange')

        plt.title(f'Price Prediction for {os.path.splitext(stock_filename)[0]}')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.show()

