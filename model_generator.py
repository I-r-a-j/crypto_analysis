import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set start date for data and today's date
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Fixed prediction period of 5 days
PREDICTION_PERIOD = 5

# Define a list of crypto symbols
CRYPTO_SYMBOLS = ['BTC-USD', 'ETH-USD', 'LTC-USD', 'DOGE-USD']

# Define models directory path - ensure this matches your GitHub repo structure
MODELS_DIR = "models"

def ensure_directory_exists(directory):
    """Ensure the directory exists, creating it if necessary."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def load_data(ticker):
    """Download historical data for the given ticker."""
    logger.info(f"Downloading data for {ticker} from {START} to {TODAY}")
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

def prepare_features(data):
    """Prepare features for the ML model."""
    # Find the correct 'Close' column name
    close_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
    
    # Prepare data for forecasting
    df = pd.DataFrame()
    df['ds'] = data['Date']
    df['y'] = data[close_col]
    
    # Feature Engineering
    df['SMA_10'] = df['y'].rolling(window=10).mean()
    df['SMA_30'] = df['y'].rolling(window=30).mean()
    df['EMA_10'] = df['y'].ewm(span=10, adjust=False).mean()
    df['EMA_30'] = df['y'].ewm(span=30, adjust=False).mean()
    
    # Add volatility features
    df['volatility_10'] = df['y'].rolling(window=10).std()
    df['volatility_30'] = df['y'].rolling(window=30).std()
    
    # Add price momentum
    df['price_change_1d'] = df['y'].pct_change(periods=1)
    df['price_change_7d'] = df['y'].pct_change(periods=7)
    
    # Add date-based features
    df['day'] = pd.to_datetime(df['ds']).dt.day
    df['month'] = pd.to_datetime(df['ds']).dt.month
    df['year'] = pd.to_datetime(df['ds']).dt.year
    df['day_of_week'] = pd.to_datetime(df['ds']).dt.dayofweek
    
    # Drop rows with NaN values
    df = df.dropna()
    
    return df

def generate_future_features(last_known_data, future_dates, feature_columns):
    """Generate features for future prediction dates."""
    n_dates = len(future_dates)
    
    future_features = pd.DataFrame({
        'day': [d.day for d in future_dates],
        'month': [d.month for d in future_dates],
        'year': [d.year for d in future_dates],
        'day_of_week': [d.dayofweek for d in future_dates]
    })
    
    # Add technical indicators from last known data
    for col in feature_columns:
        if col not in future_features.columns:
            future_features[col] = last_known_data[col]
    
    # Ensure correct column order
    return future_features[feature_columns]

def train_and_save_model(symbol):
    """Train ML model for a crypto symbol and save it."""
    try:
        # Load data for the symbol
        data = load_data(symbol)
        
        # Prepare features
        df_train = prepare_features(data)
        
        # Define feature columns (excluding date and target)
        feature_columns = [
            'SMA_10', 'SMA_30', 'EMA_10', 'EMA_30', 
            'volatility_10', 'volatility_30',
            'price_change_1d', 'price_change_7d',
            'day', 'month', 'year', 'day_of_week'
        ]
        
        # Train-test split
        X = df_train[feature_columns]
        y = df_train['y']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Define and train the Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Calculate model performance metrics
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        logger.info(f"{symbol} model performance - Train R²: {train_score:.4f}, Test R²: {test_score:.4f}")
        
        # Prepare the directory where models will be saved
        ensure_directory_exists(MODELS_DIR)
        
        # Save the model
        clean_symbol = symbol.replace('-', '_')
        model_filename = os.path.join(MODELS_DIR, f'{clean_symbol}_model.pkl')
        joblib.dump(model, model_filename)
        logger.info(f"Trained model for {symbol} saved to {model_filename}")
        
        # Generate predictions for future dates
        last_known_data = df_train.iloc[-1]
        future_dates = pd.date_range(
            df_train['ds'].iloc[-1] + pd.Timedelta(days=1),
            periods=PREDICTION_PERIOD, 
            freq='D'
        )
        
        future_features = generate_future_features(df_train, future_dates, feature_columns)
        future_close = model.predict(future_features)
        
        # Create predictions DataFrame
        future_df = pd.DataFrame({
            'Date': future_dates, 
            'Predicted_Close': future_close
        })
        
        # Save predictions
        predictions_filename = os.path.join(MODELS_DIR, f'{clean_symbol}_predictions.csv')
        future_df.to_csv(predictions_filename, index=False)
        logger.info(f"Future predictions for {symbol} saved to {predictions_filename}")
        
        # Save model metadata
        metadata = {
            'symbol': symbol,
            'training_data_start': START,
            'training_data_end': TODAY,
            'model_created': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'features': feature_columns,
            'train_score': train_score,
            'test_score': test_score
        }
        
        metadata_filename = os.path.join(MODELS_DIR, f'{clean_symbol}_metadata.json')
        pd.Series(metadata).to_json(metadata_filename)
        logger.info(f"Model metadata for {symbol} saved to {metadata_filename}")
        
        return model, future_df
        
    except Exception as e:
        logger.error(f"Error training model for {symbol}: {str(e)}")
        raise

def main():
    """Main function to train and save models for all symbols."""
    logger.info(f"Starting cryptocurrency model training process at {datetime.now()}")
    
    # Ensure models directory exists
    ensure_directory_exists(MODELS_DIR)
    
    # Train models for all symbols
    for symbol in CRYPTO_SYMBOLS:
        logger.info(f"Processing {symbol}...")
        try:
            train_and_save_model(symbol)
        except Exception as e:
            logger.error(f"Failed to process {symbol}: {e}")
    
    logger.info(f"Completed cryptocurrency model training process at {datetime.now()}")

if __name__ == "__main__":
    main()
