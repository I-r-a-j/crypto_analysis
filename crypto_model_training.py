import os
from datetime import date, datetime, timedelta
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import logging
from pathlib import Path

# Configuration
START_DATE = "2015-01-01"
PREDICTION_DAYS = 5
CRYPTO_SYMBOLS = ['BTC-USD', 'ETH-USD', 'LTC-USD', 'DOGE-USD']
MODELS_DIR = "models_with_predictions"
LOG_FILE = "crypto_predictions.log"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

def create_directories():
    """Create required directories if they don't exist"""
    Path(MODELS_DIR).mkdir(exist_ok=True)

def load_data(ticker):
    """Load historical data for a given ticker"""
    try:
        data = yf.download(ticker, START_DATE, date.today().strftime("%Y-%m-%d"))
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        logging.error(f"Error loading data for {ticker}: {str(e)}")
        return None

def prepare_features(data):
    """Prepare features for the model"""
    close_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
    
    df = pd.DataFrame()
    df['ds'] = data['Date']
    df['y'] = data[close_col]
    
    # Technical indicators
    df['SMA_10'] = df['y'].rolling(window=10).mean()
    df['SMA_30'] = df['y'].rolling(window=30).mean()
    df['EMA_10'] = df['y'].ewm(span=10, adjust=False).mean()
    df['EMA_30'] = df['y'].ewm(span=30, adjust=False).mean()
    
    # Date features
    df['day'] = pd.to_datetime(df['ds']).dt.day
    df['month'] = pd.to_datetime(df['ds']).dt.month
    df['year'] = pd.to_datetime(df['ds']).dt.year
    
    return df.dropna()

def train_model(X_train, y_train):
    """Train and return a Linear Regression model"""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def generate_future_features(last_known_data, future_dates, feature_columns):
    """Generate features for future prediction dates"""
    n_dates = len(future_dates)
    return pd.DataFrame({
        'day': [d.day for d in future_dates],
        'month': [d.month for d in future_dates],
        'year': [d.year for d in future_dates],
        'SMA_10': [last_known_data['SMA_10']] * n_dates,
        'SMA_30': [last_known_data['SMA_30']] * n_dates,
        'EMA_10': [last_known_data['EMA_10']] * n_dates,
        'EMA_30': [last_known_data['EMA_30']] * n_dates
    })[feature_columns]

def make_predictions(model, last_known_data, feature_columns):
    """Generate predictions for the next PREDICTION_DAYS days"""
    last_date = pd.to_datetime(last_known_data['ds'])
    future_dates = [last_date + timedelta(days=i) for i in range(1, PREDICTION_DAYS + 1)]
    
    future_features = generate_future_features(last_known_data, future_dates, feature_columns)
    future_prices = model.predict(future_features)
    
    predictions = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Close': future_prices,
        'Prediction_Date': datetime.now().strftime("%Y-%m-%d")
    })
    predictions.set_index('Date', inplace=True)
    
    return predictions

def save_model_with_predictions(model, predictions, symbol):
    """Save trained model with integrated predictions as a single file"""
    model_data = {
        'model': model,
        'predictions': predictions,
        'last_trained': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'prediction_days': PREDICTION_DAYS,
        'symbol': symbol
    }
    
    model_path = os.path.join(MODELS_DIR, f"{symbol}_model_with_predictions.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    logging.info(f"Model with predictions saved to {model_path}")

def process_crypto(symbol):
    """Full processing pipeline for a single cryptocurrency"""
    logging.info(f"Processing {symbol}...")
    
    # Load and prepare data
    data = load_data(symbol)
    if data is None:
        return
        
    df = prepare_features(data)
    if df.empty:
        logging.warning(f"No data available for {symbol} after feature preparation")
        return
    
    # Split data and train model
    feature_columns = ['SMA_10', 'SMA_30', 'EMA_10', 'EMA_30', 'day', 'month', 'year']
    X = df[feature_columns]
    y = df['y']
    
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    
    # Generate predictions
    last_known_data = df.iloc[-1]
    predictions = make_predictions(model, last_known_data, feature_columns)
    
    # Save model with integrated predictions
    save_model_with_predictions(model, predictions, symbol)
    
    logging.info(f"Completed processing for {symbol}\n")

def main():
    """Main execution function"""
    logging.info("Starting crypto prediction pipeline")
    create_directories()
    
    for symbol in CRYPTO_SYMBOLS:
        process_crypto(symbol)
    
    logging.info("Prediction pipeline completed")

if __name__ == "__main__":
    main()
