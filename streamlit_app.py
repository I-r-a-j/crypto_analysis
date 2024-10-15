import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import requests
import pickle
from datetime import date, timedelta
from pycoingecko import CoinGeckoAPI

# Initialize CoinGecko API client
cg = CoinGeckoAPI()

# Google Drive links for the models
MODEL_URLS = {
    'Bitcoin (BTC)': "https://drive.google.com/uc?export=download&id=1-55iPtncWPsMzxDOHOsLNbv0snuQUTcJ",
    'Ethereum (ETH)': "https://drive.google.com/uc?export=download&id=1-7QoFQThAweJnxmixjKSazFQXyN0FAU_",
    'Litecoin (LTC)': "https://drive.google.com/uc?export=download&id=1-Ajon8ebaYzuI-TDLj14UziC0meqVTc-",
    'Dogecoin (DOGE)': "https://drive.google.com/uc?export=download&id=1-RC4K3aC7eqtrifKOZicRgE6RLJpsk7G"
}

# Cache the model to avoid re-downloading
@st.cache_resource
def load_model(url):
    response = requests.get(url)
    with open("crypto_model.pkl", "wb") as file:
        file.write(response.content)
    
    with open("crypto_model.pkl", "rb") as file:
        model = pickle.load(file)
    
    return model

# Function to fetch data from Yahoo Finance
def fetch_data(symbol, period='5y'):
    df = yf.download(symbol, period=period, progress=False)
    df.reset_index(inplace=True)
    return df

# Function to plot candlestick chart
def plot_candlestick_chart(df):
    fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                                         open=df['Open'],
                                         high=df['High'],
                                         low=df['Low'],
                                         close=df['Close'])])
    fig.update_layout(title='Candlestick Chart', xaxis_title='Date', yaxis_title='Price')
    return fig

# Perform technical analysis function
def perform_technical_analysis(df, analysis_type):
    # ... (Same as in the original script)
    return fig, recommendation

# Function to load cryptocurrency data from CoinGecko (past 365 days)
def load_data(coin):
    market_data = cg.get_coin_market_chart_by_id(id=coin, vs_currency='usd', days=365)
    prices = market_data['prices']
    
    # Convert to DataFrame
    data = pd.DataFrame(prices, columns=['timestamp', 'Close'])
    data['Date'] = pd.to_datetime(data['timestamp'], unit='ms')
    data = data[['Date', 'Close']]
    
    return data

# Constants for the prediction model
START = (date.today() - timedelta(days=365)).strftime("%Y-%m-%d")
TODAY = date.today().strftime("%Y-%m-%d")
period = 5

# Streamlit app
st.title("Cryptocurrency Dashboard & Prediction")

# Sidebar options
st.sidebar.title("Options")
symbols = ['btc-usd', 'eth-usd', 'ltc-usd', 'doge-usd']
crypto_options = {
    'Bitcoin (BTC)': 'bitcoin',
    'Ethereum (ETH)': 'ethereum',
    'Litecoin (LTC)': 'litecoin',
    'Dogecoin (DOGE)': 'dogecoin'
}
selected_crypto = st.sidebar.selectbox('Select Cryptocurrency', list(crypto_options.keys()))
selected_coin = crypto_options[selected_crypto]

# Fetch data for selected symbol
data = fetch_data(selected_coin)
# Display candlestick chart
st.subheader(f"{selected_crypto.upper()} Candlestick Chart")
st.plotly_chart(plot_candlestick_chart(data))

# Technical analysis
technical_analysis_type = st.sidebar.selectbox('Select Technical Analysis Type', 
                                               ['Moving Averages', 'RSI', 'MACD', 'Bollinger Bands', 
                                                'Exponential Moving Averages (EMA)', 
                                                'Stochastic Oscillator', 
                                                'Ichimoku Cloud'])

st.subheader(f"{selected_crypto.upper()} {technical_analysis_type} Analysis")
analysis_fig, recommendation = perform_technical_analysis(data, technical_analysis_type)
st.plotly_chart(analysis_fig)
st.markdown(recommendation)

# Load model corresponding to selected cryptocurrency
MODEL_URL = MODEL_URLS[selected_crypto]
model = load_model(MODEL_URL)

# Load data for prediction
data = load_data(selected_coin)

# Prepare the data for predictions
df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})

# Feature Engineering
df_train['SMA_10'] = df_train['y'].rolling(window=10).mean()
df_train['SMA_30'] = df_train['y'].rolling(window=30).mean()
df_train['EMA_10'] = df_train['y'].ewm(span=10, adjust=False).mean()
df_train['EMA_30'] = df_train['y'].ewm(span=30, adjust=False).mean()
df_train['day'] = df_train['ds'].dt.day
df_train['month'] = df_train['ds'].dt.month
df_train['year'] = df_train['ds'].dt.year

# Drop rows with NaN values
df_train = df_train.dropna()

# Ensure feature order matches the training data
features_order = ['SMA_10', 'SMA_30', 'EMA_10', 'EMA_30', 'day', 'month', 'year']

# Show raw data (only the last 10 rows)
st.subheader(f"Raw Data for {selected_crypto}")
st.write(data.iloc[:-1].tail(10))

# Prepare future features for prediction
today = pd.Timestamp(TODAY)
future_dates = pd.date_range(today, periods=period, freq='D').tolist()
last_row = df_train.tail(1)
future_features = pd.DataFrame({
    'day': [d.day for d in future_dates],
    'month': [d.month for d in future_dates],
    'year': [d.year for d in future_dates],
    'SMA_10': last_row['SMA_10'].values[0],
    'SMA_30': last_row['SMA_30'].values[0],
    'EMA_10': last_row['EMA_10'].values[0],
    'EMA_30': last_row['EMA_30'].values[0]
})

# Prediction logic (use the model)
# (Assuming a prediction step exists in your original model)

# Display prediction results (after completing the prediction logic)
st.subheader(f"Predicted Prices for the Next 5 Days ({selected_crypto})")
# st.write(predictions)  # Uncomment after adding prediction logic
