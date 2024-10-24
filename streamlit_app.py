import requests
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import date, timedelta
from pycoingecko import CoinGeckoAPI

# Initialize CoinGecko API client
cg = CoinGeckoAPI()

# Google Drive links for the models
MODEL_URLS = {
    'Bitcoin (BTC)': "https://drive.google.com/uc?export=download&id=15fpJ48AGZqoXSIr3kQHnCLvsg_be8D5r",
    'Ethereum (ETH)': "https://drive.google.com/uc?export=download&id=1q5I7bwqVI8_J28HXPx4DwzqHjtdTSib0",
    'Litecoin (LTC)': "https://drive.google.com/uc?export=download&id=1ay0MI8xoA2HGvhjDyRrhBoaTqSaLx7zv",
    'Dogecoin (DOGE)': "https://drive.google.com/uc?export=download&id=1ImJH3OsLPGlgDsEyg1Hllih0J-T29WuC"
}

# Function to download the model from Google Drive
@st.cache_resource
def load_model(url):
    response = requests.get(url)
    with open("crypto_model.pkl", "wb") as file:
        file.write(response.content)
    
    with open("crypto_model.pkl", "rb") as file:
        model = pickle.load(file)
    
    return model

# Constants
START = (date.today() - timedelta(days=365)).strftime("%Y-%m-%d")  # Past year
TODAY = date.today().strftime("%Y-%m-%d")
period = 5  # Predicting for the next 5 days

# Streamlit UI
st.title("Cryptocurrency Price Prediction & Technical Analysis")

# Dropdown for selecting cryptocurrency
crypto_options = {
    'Bitcoin (BTC)': 'bitcoin',
    'Ethereum (ETH)': 'ethereum',
    'Litecoin (LTC)': 'litecoin',
    'Dogecoin (DOGE)': 'dogecoin'
}
selected_crypto = st.selectbox('Select Cryptocurrency', list(crypto_options.keys()))
selected_coin = crypto_options[selected_crypto]

# Load the model corresponding to the selected cryptocurrency
MODEL_URL = MODEL_URLS[selected_crypto]
model = load_model(MODEL_URL)

# Function to load cryptocurrency data from CoinGecko (past 365 days)
def load_data(coin):
    market_data = cg.get_coin_market_chart_by_id(id=coin, vs_currency='usd', days=365)
    prices = market_data['prices']
    
    # Convert to DataFrame
    data = pd.DataFrame(prices, columns=['timestamp', 'Close'])
    data['Date'] = pd.to_datetime(data['timestamp'], unit='ms')  # Convert from timestamp to date
    data = data[['Date', 'Close']]
    
    return data

# Load the data
data = load_data(selected_coin)

# Prepare the data for predictions
df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})

# Show raw data (only the last 10 rows, excluding the last row)
st.subheader(f"Raw Data for {selected_crypto}")
st.write(data.iloc[:-1].tail(10))  # Exclude the last row and show the last 10 remaining rows

# Feature Engineering (Simple Moving Average, Exponential Moving Average)
df_train['SMA_10'] = df_train['y'].rolling(window=10).mean()  # 10-day Simple Moving Average
df_train['SMA_30'] = df_train['y'].rolling(window=30).mean()  # 30-day Simple Moving Average
df_train['EMA_10'] = df_train['y'].ewm(span=10, adjust=False).mean()  # 10-day Exponential Moving Average
df_train['EMA_30'] = df_train['y'].ewm(span=30, adjust=False).mean()  # 30-day Exponential Moving Average

# RSI Calculation
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, abs(delta), 0)
    avg_gain = pd.Series(gain).rolling(window=window, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df_train['RSI'] = calculate_rsi(df_train['y'])

# MACD Calculation
def calculate_macd(data, slow=26, fast=12, signal=9):
    ema_fast = data.ewm(span=fast, adjust=False).mean()
    ema_slow = data.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_histogram = macd - macd_signal
    return macd, macd_signal, macd_histogram

df_train['MACD'], df_train['MACD_Signal'], df_train['MACD_Histogram'] = calculate_macd(df_train['y'])

# Bollinger Bands Calculation
def calculate_bollinger_bands(data, window=20, std_factor=2):
    sma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    upper_band = sma + (std_factor * std)
    lower_band = sma - (std_factor * std)
    return upper_band, lower_band

df_train['Bollinger_Upper'], df_train['Bollinger_Lower'] = calculate_bollinger_bands(df_train['y'])

# Recommendations based on indicators
def get_recommendation(df):
    recommendations = {}

    # Simple Moving Averages (SMA)
    if df['SMA_10'].iloc[-1] > df['SMA_30'].iloc[-1]:
        recommendations['SMA'] = 'Buy'
    else:
        recommendations['SMA'] = 'Sell'

    # Exponential Moving Averages (EMA)
    if df['EMA_10'].iloc[-1] > df['EMA_30'].iloc[-1]:
        recommendations['EMA'] = 'Buy'
    else:
        recommendations['EMA'] = 'Sell'

    # RSI
    if df['RSI'].iloc[-1] < 30:
        recommendations['RSI'] = 'Buy (Oversold)'
    elif df['RSI'].iloc[-1] > 70:
        recommendations['RSI'] = 'Sell (Overbought)'
    else:
        recommendations['RSI'] = 'Neutral'

    # MACD
    if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]:
        recommendations['MACD'] = 'Buy'
    else:
        recommendations['MACD'] = 'Sell'

    return recommendations

# Get recommendations
recommendations = get_recommendation(df_train)

# Display recommendations
st.subheader(f"Technical Analysis Recommendations for {selected_crypto}")
for indicator, recommendation in recommendations.items():
    st.write(f"{indicator}: {recommendation}")

# Plotting
st.subheader(f"Technical Indicator Plots for {selected_crypto}")

# Candlestick Chart
fig = go.Figure(data=[go.Candlestick(
    x=df_train['ds'],
    open=df_train['y'].shift(1),
    high=df_train['y'].rolling(window=3).max(),
    low=df_train['y'].rolling(window=3).min(),
    close=df_train['y'],
    increasing_line_color='green', decreasing_line_color='red'
)])

# Plot Moving Averages
fig.add_trace(go.Scatter(x=df_train['ds'], y=df_train['SMA_10'], mode='lines', name='SMA 10'))
fig.add_trace(go.Scatter(x=df_train['ds'], y=df_train['SMA_30'], mode='lines', name='SMA 30'))
fig.add_trace(go.Scatter(x=df_train['ds'], y=df_train['EMA_10'], mode='lines', name='EMA 10'))
fig.add_trace(go.Scatter(x=df_train['ds'], y=df_train['EMA_30'], mode='lines', name='EMA 30'))

# Plot Bollinger Bands
fig.add_trace(go.Scatter(x=df_train['ds'], y=df_train['Bollinger_Upper'], mode='lines', name='Bollinger Upper'))
fig.add_trace(go.Scatter(x=df_train['ds'], y=df_train['Bollinger_Lower'], mode='lines', name='Bollinger Lower'))

st.plotly_chart(fig)

# Plot RSI
st.subheader(f"RSI Plot for {selected_crypto}")
fig_rsi = go.Figure(data=[go.Scatter(x=df_train['ds'], y=df_train['RSI'], mode='lines', name='RSI')])
fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")  # Oversold threshold
fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")  # Overbought threshold
st.plotly_chart(fig_rsi)

# Plot MACD
st.subheader(f"MACD Plot for {selected_crypto}")
fig_macd = go.Figure()
fig_macd.add_trace(go.Scatter(x=df_train['ds'], y=df_train['MACD'], mode='lines', name='MACD'))
fig_macd.add_trace(go.Scatter(x=df_train['ds'], y=df_train['MACD_Signal'], mode='lines', name='MACD Signal'))
fig_macd.add_trace(go.Bar(x=df_train['ds'], y=df_train['MACD_Histogram'], name='MACD Histogram'))
st.plotly_chart(fig_macd)
