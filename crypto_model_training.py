from datetime import date
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Set start date for data and today's date
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Fixed prediction period of 5 days
period = 5

# Define a list of crypto symbols (BTC, ETH, LTC, DOGE)
crypto_symbols = ['BTC-USD', 'ETH-USD', 'LTC-USD', 'DOGE-USD']

def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    # Reset index to get Date as a column
    data.reset_index(inplace=True)
    return data

def train_and_save_model(symbol):
    # Load data for the symbol
    data = load_data(symbol)

    # Find the correct 'Close' column name
    close_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'

    # Prepare data for forecasting
    df_train = pd.DataFrame()
    df_train['ds'] = data['Date']
    df_train['y'] = data[close_col]

    # Feature Engineering
    df_train['SMA_10'] = df_train['y'].rolling(window=10).mean()
    df_train['SMA_30'] = df_train['y'].rolling(window=30).mean()
    df_train['EMA_10'] = df_train['y'].ewm(span=10, adjust=False).mean()
    df_train['EMA_30'] = df_train['y'].ewm(span=30, adjust=False).mean()

    # Add date-based features
    df_train['day'] = pd.to_datetime(df_train['ds']).dt.day
    df_train['month'] = pd.to_datetime(df_train['ds']).dt.month
    df_train['year'] = pd.to_datetime(df_train['ds']).dt.year

    # Drop rows with NaN values
    df_train = df_train.dropna()

    # Define feature columns
    feature_columns = ['SMA_10', 'SMA_30', 'EMA_10', 'EMA_30', 'day', 'month', 'year']

    # Train-test split
    X = df_train[feature_columns]
    y = df_train['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the Linear Regression model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Save the model
    model_filename = f'{symbol}_linear_regression_model.pkl'
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Trained model for {symbol} saved as {model_filename}")

    def generate_future_features(last_known_data, future_dates):
        # Create lists of repeated values for the moving averages
        n_dates = len(future_dates)
        future_features = pd.DataFrame({
            'day': [d.day for d in future_dates],
            'month': [d.month for d in future_dates],
            'year': [d.year for d in future_dates],
            'SMA_10': [last_known_data['SMA_10']] * n_dates,
            'SMA_30': [last_known_data['SMA_30']] * n_dates,
            'EMA_10': [last_known_data['EMA_10']] * n_dates,
            'EMA_30': [last_known_data['EMA_30']] * n_dates
        })
        return future_features

    # Predicting future data
    last_known_data = df_train.iloc[-1]
    future_dates = pd.date_range(df_train['ds'].iloc[-1] + pd.Timedelta(days=1),
                                periods=period, freq='D')
    future_features = generate_future_features(last_known_data, future_dates)

    # Ensure correct column order
    future_features = future_features[feature_columns]

    # Predict future prices
    future_close = model.predict(future_features)

    # Create predictions DataFrame
    future_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_close})
    future_df.set_index('Date', inplace=True)

    print(f"Forecast data (Next 5 days) for {symbol}:")
    print(future_df)

# Train models for all symbols
for symbol in crypto_symbols:
    train_and_save_model(symbol)
