import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import plotly.graph_objects as go
from datetime import date, timedelta
from pycoingecko import CoinGeckoAPI

# Initialize CoinGecko API client
cg = CoinGeckoAPI()

# Constants
START = (date.today() - timedelta(days=365)).strftime("%Y-%m-%d")
TODAY = date.today().strftime("%Y-%m-%d")
PREDICTION_PERIOD = 5  # Predicting for the next 5 days

# Google Drive links for the models
MODEL_URLS = {
    'Bitcoin (BTC)': "https://drive.google.com/uc?export=download&id=15fpJ48AGZqoXSIr3kQHnCLvsg_be8D5r",
    'Ethereum (ETH)': "https://drive.google.com/uc?export=download&id=1q5I7bwqVI8_J28HXPx4DwzqHjtdTSib0",
    'Litecoin (LTC)': "https://drive.google.com/uc?export=download&id=1ay0MI8xoA2HGvhjDyRrhBoaTqSaLx7zv",
    'Dogecoin (DOGE)': "https://drive.google.com/uc?export=download&id=1ImJH3OsLPGlgDsEyg1Hllih0J-T29WuC"
}

# Cryptocurrency options
CRYPTO_OPTIONS = {
    'Bitcoin (BTC)': 'bitcoin',
    'Ethereum (ETH)': 'ethereum',
    'Litecoin (LTC)': 'litecoin',
    'Dogecoin (DOGE)': 'dogecoin'
}

def load_data(coin):
    """Fetch historical price data from CoinGecko"""
    market_data = cg.get_coin_market_chart_by_id(id=coin, vs_currency='usd', days=365)
    prices = market_data['prices']
    data = pd.DataFrame(prices, columns=['timestamp', 'Close'])
    data['Date'] = pd.to_datetime(data['timestamp'], unit='ms')
    data = data[['Date', 'Close']]
    return data

def moving_average(data, window):
    """Calculate Moving Average"""
    return data['Close'].rolling(window=window).mean()

def rsi(data, window=14):
    """Calculate Relative Strength Index"""
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def macd(data, short_window=12, long_window=26, signal_window=9):
    """Calculate MACD"""
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    return macd_line, signal_line

def bollinger_bands(data, window=20):
    """Calculate Bollinger Bands"""
    sma = data['Close'].rolling(window=window).mean()
    std_dev = data['Close'].rolling(window=window).std()
    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)
    return upper_band, lower_band

def ema(data, window):
    """Calculate Exponential Moving Average"""
    return data['Close'].ewm(span=window, adjust=False).mean()

@st.cache_resource
def load_model(url):
    """Download and load the model from Google Drive"""
    response = requests.get(url)
    with open("crypto_model.pkl", "wb") as file:
        file.write(response.content)
    
    with open("crypto_model.pkl", "rb") as file:
        model = pickle.load(file)
    
    return model

def prepare_prediction_features(data):
    """Prepare features for prediction"""
    df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
    
    # Feature Engineering
    df_train['SMA_10'] = df_train['y'].rolling(window=10).mean()
    df_train['SMA_30'] = df_train['y'].rolling(window=30).mean()
    df_train['EMA_10'] = df_train['y'].ewm(span=10, adjust=False).mean()
    df_train['EMA_30'] = df_train['y'].ewm(span=30, adjust=False).mean()
    
    df_train['day'] = df_train['ds'].dt.day
    df_train['month'] = df_train['ds'].dt.month
    df_train['year'] = df_train['ds'].dt.year
    
    return df_train.dropna()

def main():
    st.title("Cryptocurrency Price Prediction and Technical Analysis")
    
    # Select cryptocurrency
    selected_crypto = st.selectbox('Select Cryptocurrency', list(CRYPTO_OPTIONS.keys()))
    selected_coin = CRYPTO_OPTIONS[selected_crypto]
    
    # Load data
    data = load_data(selected_coin)
    
    # Display candlestick chart
    st.subheader(f"Candlestick Chart for {selected_crypto}")
    fig = go.Figure(data=[go.Candlestick(
        x=data['Date'],
        open=data['Close'],
        high=data['Close'] * 1.01,
        low=data['Close'] * 0.99,
        close=data['Close']
    )])
    
    # Technical indicator selection
    indicator = st.selectbox(
        'Select Technical Indicator',
        ['Moving Average', 'RSI', 'MACD', 'Bollinger Bands', 'EMA']
    )
    
    # Handle technical indicators
    if indicator == 'Moving Average':
        window = st.slider('Select MA Window (days)', 5, 100, 20)
        ma = moving_average(data, window)
        fig.add_trace(go.Scatter(x=data['Date'], y=ma, mode='lines', name=f'MA {window} days'))
        st.write(f"**Recommendation:** Look for crossovers between the price and MA for trend signals.")
    
    elif indicator == 'RSI':
        window = st.slider('Select RSI Window (days)', 5, 100, 14)
        rsi_values = rsi(data, window)
        fig.add_trace(go.Scatter(x=data['Date'], y=rsi_values, mode='lines', name=f'RSI {window} days'))
        st.write("**Recommendation:** RSI above 70 suggests overbought, below 30 suggests oversold.")
    
    elif indicator == 'MACD':
        macd_line, signal_line = macd(data)
        fig.add_trace(go.Scatter(x=data['Date'], y=macd_line, mode='lines', name='MACD Line'))
        fig.add_trace(go.Scatter(x=data['Date'], y=signal_line, mode='lines', name='Signal Line'))
        st.write("**Recommendation:** MACD crossing above signal line is bullish, below is bearish.")
    
    elif indicator == 'Bollinger Bands':
        upper_band, lower_band = bollinger_bands(data)
        fig.add_trace(go.Scatter(x=data['Date'], y=upper_band, mode='lines', name='Upper Band'))
        fig.add_trace(go.Scatter(x=data['Date'], y=lower_band, mode='lines', name='Lower Band'))
        st.write("**Recommendation:** Price at bands suggests potential reversal points.")
    
    elif indicator == 'EMA':
        window = st.slider('Select EMA Window (days)', 5, 100, 20)
        ema_values = ema(data, window)
        fig.add_trace(go.Scatter(x=data['Date'], y=ema_values, mode='lines', name=f'EMA {window} days'))
        st.write("**Recommendation:** Price above EMA suggests bullish, below suggests bearish.")
    
    st.plotly_chart(fig)
    
    # Price Prediction Section
    st.title("Price Prediction (Next 5 Days)")
    
    # Load and prepare model
    model = load_model(MODEL_URLS[selected_crypto])
    df_train = prepare_prediction_features(data)
    
    # Prepare future features
    future_dates = pd.date_range(TODAY, periods=PREDICTION_PERIOD, freq='D')
    last_row = df_train.tail(1)
    
    future_features = pd.DataFrame({
        'day': [d.day for d in future_dates],
        'month': [d.month for d in future_dates],
        'year': [d.year for d in future_dates],
        'SMA_10': [last_row['SMA_10'].values[0]] * PREDICTION_PERIOD,
        'SMA_30': [last_row['SMA_30'].values[0]] * PREDICTION_PERIOD,
        'EMA_10': [last_row['EMA_10'].values[0]] * PREDICTION_PERIOD,
        'EMA_30': [last_row['EMA_30'].values[0]] * PREDICTION_PERIOD
    })
    
    # Make predictions
    features_order = ['SMA_10', 'SMA_30', 'EMA_10', 'EMA_30', 'day', 'month', 'year']
    future_features = future_features[features_order]
    future_close = model.predict(future_features)
    
    # Display predictions
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Close': future_close
    })
    future_df.set_index('Date', inplace=True)
    st.write(future_df)

if __name__ == "__main__":
    main()
