import requests
import pickle
import streamlit as st
import pandas as pd
from datetime import date, timedelta
from pycoingecko import CoinGeckoAPI  # Import CoinGeckoAPI for cryptocurrency data

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
@st.cache_resource  # Cache the model to avoid re-downloading
def load_model(url):
    response = requests.get(url)
    with open("crypto_model.pkl", "wb") as file:
        file.write(response.content)
    
    with open("crypto_model.pkl", "rb") as file:
        model = pickle.load(file)
    
    return model

# Constants
START = (date.today() - timedelta(days=365)).strftime("%Y-%m-%d")  # Restrict to past year
TODAY = date.today().strftime("%Y-%m-%d")
period = 5  # Predicting for the next 5 days

# Streamlit UI
st.title("Cryptocurrency Price Prediction (Next 5 Days)")

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

# Feature Engineering
df_train['SMA_10'] = df_train['y'].rolling(window=10).mean()  # 10-day Simple Moving Average
df_train['SMA_30'] = df_train['y'].rolling(window=30).mean()  # 30-day Simple Moving Average
df_train['EMA_10'] = df_train['y'].ewm(span=10, adjust=False).mean()  # 10-day Exponential Moving Average
df_train['EMA_30'] = df_train['y'].ewm(span=30, adjust=False).mean()  # 30-day Exponential Moving Average

df_train['day'] = df_train['ds'].dt.day
df_train['month'] = df_train['ds'].dt.month
df_train['year'] = df_train['ds'].dt.year

# Drop rows with NaN values
df_train = df_train.dropna()

# Ensure feature order matches the training data
features_order = ['SMA_10', 'SMA_30', 'EMA_10', 'EMA_30', 'day', 'month', 'year']

# Show raw data (only the last 10 rows, excluding the last row)
st.subheader(f"Raw Data for {selected_crypto}")
st.write(data.iloc[:-1].tail(10))  # Exclude the last row and show the last 10 remaining rows

# Prepare future features for prediction
today = pd.Timestamp(TODAY)

# Generate future dates starting from today (current date)
future_dates = pd.date_range(today, periods=period, freq='D').tolist()

# Use the most recent feature data for predictions (from the last row of df_train)
last_row = df_train.tail(1)

# Generate new feature data for future dates (moving averages and date features)
future_features = pd.DataFrame({
    'day': [d.day for d in future_dates],
    'month': [d.month for d in future_dates],
    'year': [d.year for d in future_dates],
    'SMA_10': last_row['SMA_10'].values[0],  # Last known SMA_10 value
    'SMA_30': last_row['SMA_30'].values[0],  # Last known SMA_30 value
    'EMA_10': last_row['EMA_10'].values[0],  # Last known EMA_10 value
    'EMA_30': last_row['EMA_30'].values[0]   # Last known EMA_30 value
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
