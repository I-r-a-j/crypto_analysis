import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import date, timedelta
from pycoingecko import CoinGeckoAPI

# Initialize CoinGecko API client
cg = CoinGeckoAPI()

# Function to fetch historical price data
def load_data(coin):
    market_data = cg.get_coin_market_chart_by_id(id=coin, vs_currency='usd', days=365)
    prices = market_data['prices']
    data = pd.DataFrame(prices, columns=['timestamp', 'Close'])
    data['Date'] = pd.to_datetime(data['timestamp'], unit='ms')
    data = data[['Date', 'Close']]
    return data

# Function to calculate Moving Averages (MA)
def moving_average(data, window):
    return data['Close'].rolling(window=window).mean()

# Function to calculate Relative Strength Index (RSI)
def rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate Moving Average Convergence Divergence (MACD)
def macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    return macd_line, signal_line

# Function to calculate Bollinger Bands
def bollinger_bands(data, window=20):
    sma = data['Close'].rolling(window=window).mean()
    std_dev = data['Close'].rolling(window=window).std()
    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)
    return upper_band, lower_band

# Function to calculate Exponential Moving Average (EMA)
def ema(data, window):
    return data['Close'].ewm(span=window, adjust=False).mean()

# Streamlit UI
st.title("Cryptocurrency Price Prediction and Technical Analysis")

# Dropdown for selecting cryptocurrency
crypto_options = {
    'Bitcoin (BTC)': 'bitcoin',
    'Ethereum (ETH)': 'ethereum',
    'Litecoin (LTC)': 'litecoin',
    'Dogecoin (DOGE)': 'dogecoin'
}
selected_crypto = st.selectbox('Select Cryptocurrency', list(crypto_options.keys()))
selected_coin = crypto_options[selected_crypto]

# Load cryptocurrency data
data = load_data(selected_coin)

# Display candlestick chart
st.subheader(f"Candlestick Chart for {selected_crypto}")
fig = go.Figure(data=[go.Candlestick(x=data['Date'], open=data['Close'], high=data['Close'] * 1.01, low=data['Close'] * 0.99, close=data['Close'])])
st.plotly_chart(fig)

# Dropdown for technical indicators
indicator = st.selectbox('Select Technical Indicator', ['Moving Average', 'RSI', 'MACD', 'Bollinger Bands', 'EMA'])

# Handle each technical indicator selection
if indicator == 'Moving Average':
    window = st.slider('Select MA Window (days)', 5, 100, 20)
    ma = moving_average(data, window)
    data['MA'] = ma
    fig.add_trace(go.Scatter(x=data['Date'], y=data['MA'], mode='lines', name=f'MA {window} days'))
    st.plotly_chart(fig)
    st.write(f"**Recommendation:** Look for crossovers between the price and MA for trend signals. Above MA suggests bullishness, below MA suggests bearishness.")

elif indicator == 'RSI':
    window = st.slider('Select RSI Window (days)', 5, 100, 14)
    rsi_values = rsi(data, window)
    fig.add_trace(go.Scatter(x=data['Date'], y=rsi_values, mode='lines', name=f'RSI {window} days'))
    st.plotly_chart(fig)
    st.write(f"**Recommendation:** RSI above 70 suggests overbought conditions (sell), RSI below 30 suggests oversold conditions (buy).")

elif indicator == 'MACD':
    macd_line, signal_line = macd(data)
    fig.add_trace(go.Scatter(x=data['Date'], y=macd_line, mode='lines', name='MACD Line'))
    fig.add_trace(go.Scatter(x=data['Date'], y=signal_line, mode='lines', name='Signal Line'))
    st.plotly_chart(fig)
    st.write(f"**Recommendation:** MACD line crossing above the signal line indicates a bullish trend, crossing below indicates a bearish trend.")

elif indicator == 'Bollinger Bands':
    upper_band, lower_band = bollinger_bands(data)
    data['Upper Band'] = upper_band
    data['Lower Band'] = lower_band
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Upper Band'], mode='lines', name='Upper Band'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Lower Band'], mode='lines', name='Lower Band'))
    st.plotly_chart(fig)
    st.write(f"**Recommendation:** Price touching or crossing the upper band suggests overbought (sell), touching or crossing the lower band suggests oversold (buy).")

elif indicator == 'EMA':
    window = st.slider('Select EMA Window (days)', 5, 100, 20)
    ema_values = ema(data, window)
    data['EMA'] = ema_values
    fig.add_trace(go.Scatter(x=data['Date'], y=data['EMA'], mode='lines', name=f'EMA {window} days'))
    st.plotly_chart(fig)
    st.write(f"**Recommendation:** Price above the EMA suggests bullishness, while price below suggests bearishness.")
