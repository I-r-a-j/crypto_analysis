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
START = (date.today() - timedelta(days=365)).strftime("%Y-%m-%d")
TODAY = date.today().strftime("%Y-%m-%d")
period = 5  # Predicting for the next 5 days

# Fetch data function with progress disabled
def fetch_data(symbol, period='5y'):
    df = yf.download(symbol, period=period, progress=False)
    df.reset_index(inplace=True)
    return df

# Plot candlestick chart function
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
    # ... (keep the existing perform_technical_analysis function as is)
    # This function is quite long, so I'm not repeating it here to save space
    pass

# Function to load cryptocurrency data from CoinGecko (past 365 days)
def load_data(coin):
    market_data = cg.get_coin_market_chart_by_id(id=coin, vs_currency='usd', days=365)
    prices = market_data['prices']
    
    # Convert to DataFrame
    data = pd.DataFrame(prices, columns=['timestamp', 'Close'])
    data['Date'] = pd.to_datetime(data['timestamp'], unit='ms')
    data = data[['Date', 'Close']]
    
    return data

# Streamlit app
st.title("Cryptocurrency Analysis and Prediction Dashboard")

# Sidebar options
st.sidebar.title("Options")
crypto_options = {
    'Bitcoin (BTC)': 'btc-usd',
    'Ethereum (ETH)': 'eth-usd',
    'Litecoin (LTC)': 'ltc-usd',
    'Dogecoin (DOGE)': 'doge-usd'
}
selected_crypto = st.sidebar.selectbox('Select Cryptocurrency', list(crypto_options.keys()))
selected_symbol = crypto_options[selected_crypto]
selected_coin = selected_crypto.split()[0].lower()

technical_analysis_type = st.sidebar.selectbox('Select Technical Analysis Type', 
                                               ['Moving Averages', 'RSI', 'MACD', 'Bollinger Bands', 
                                                'Exponential Moving Averages (EMA)', 
                                                'Stochastic Oscillator', 
                                                'Ichimoku Cloud'])

# Fetch data for selected symbol
data = fetch_data(selected_symbol)

# Display candlestick chart
st.subheader(f"{selected_crypto} Candlestick Chart")
st.plotly_chart(plot_candlestick_chart(data))

# Perform selected technical analysis
st.subheader(f"{selected_crypto} {technical_analysis_type} Analysis")
analysis_fig, recommendation = perform_technical_analysis(data, technical_analysis_type)
st.plotly_chart(analysis_fig)
st.markdown(recommendation)

# Price Prediction Section
st.subheader(f"{selected_crypto} Price Prediction (Next 5 Days)")

# Load the model corresponding to the selected cryptocurrency
MODEL_URL = MODEL_URLS[selected_crypto]
model = load_model(MODEL_URL)

# Load the data for prediction
prediction_data = load_data(selected_coin)

# Prepare the data for predictions
df_train = prediction_data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})

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

# Show raw data (only the last 10 rows, excluding the last row)
st.subheader(f"Raw Data for {selected_crypto}")
st.write(prediction_data.iloc[:-1].tail(10))

# Prepare future features for prediction
today = pd.Timestamp(TODAY)
future_dates = pd.date_range(today, periods=period, freq='D').tolist()
last_row = df_train.tail(1)

# Generate new feature data for future dates
future_features = pd.DataFrame({
    'day': [d.day for d in future_dates],
    'month': [d.month for d in future_dates],
    'year': [d.year for d in future_dates],
    'SMA_10': last_row['SMA_10'].values[0],
    'SMA_30': last_row['SMA_30'].values[0],
    'EMA_10': last_row['EMA_10'].values[0],
    'EMA_30': last_row['EMA_30'].values[0]
})

# Ensure future features match the training feature order
future_features = future_features[features_order]

# Predict future prices using the pre-trained model
future_close = model.predict(future_features)

# Create a DataFrame for the predictions
future_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_close})
future_df.set_index('Date', inplace=True)

# Display the forecast data
st.subheader(f"Predicted Prices for {selected_crypto} for the Next {period} Days")
st.write(future_df)
