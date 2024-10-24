import streamlit as st
from pycoingecko import CoinGeckoAPI
import pandas as pd
import plotly.graph_objs as go
import requests
import pickle
from datetime import date, timedelta

# Initialize CoinGecko API
cg = CoinGeckoAPI()

# Dictionary mapping display names to symbols
cryptocurrency_map = {
    'Bitcoin (BTC)': 'bitcoin',
    'Ethereum (ETH)': 'ethereum',
    'Litecoin (LTC)': 'litecoin',
    'Dogecoin (DOGE)': 'dogecoin'
}

# Function to fetch cryptocurrency data
def fetch_data(crypto_name, days='365'):  # Limited to 365 days for free API users
    symbol = cryptocurrency_map[crypto_name]  # Get the symbol from the display name
    # Get market chart data for the selected cryptocurrency
    data = cg.get_coin_market_chart_by_id(id=symbol, vs_currency='usd', days=days)
    
    # Prepare the DataFrame with 'Date' and 'Close' columns
    prices = pd.DataFrame(data['prices'], columns=['Date', 'Close'])
    prices['Date'] = pd.to_datetime(prices['Date'], unit='ms')
    df = pd.DataFrame(prices)
    
    # Simulate 'Open', 'High', and 'Low' prices (since pycoingecko doesn't provide Open/High/Low)
    df['Open'] = df['Close'].shift(1)  # 'Open' is the previous day's 'Close'
    df['High'] = df['Close'].rolling(window=2).max()  # Simulate 'High' as max of last 2 days
    df['Low'] = df['Close'].rolling(window=2).min()   # Simulate 'Low' as min of last 2 days
    df.dropna(inplace=True)  # Remove rows with NaN values due to shifting

    return df

# Function to plot a candlestick chart
def plot_candlestick_chart(df):
    candlestick = go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Candlestick Chart'
    )
    
    layout = go.Layout(
        title='Candlestick Chart',
        xaxis={'title': 'Date'},
        yaxis={'title': 'Price (USD)'},
        xaxis_rangeslider_visible=False
    )
    
    fig = go.Figure(data=[candlestick], layout=layout)
    return fig

# Function to perform technical analysis
def perform_technical_analysis(df, analysis_type):
    if analysis_type == 'Moving Averages':
        df['MA50'] = df['Close'].rolling(window=50).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close Price'))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MA50'], name='50-Day MA'))
        recommendation = "Buy" if df['Close'].iloc[-1] > df['MA50'].iloc[-1] else "Sell"
    
    elif analysis_type == 'RSI':
        delta = df['Close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI'))
        recommendation = "Buy" if df['RSI'].iloc[-1] < 30 else "Sell" if df['RSI'].iloc[-1] > 70 else "Hold"
    
    elif analysis_type == 'MACD':
        df['EMA12'] = df['Close'].ewm(span=12).mean()
        df['EMA26'] = df['Close'].ewm(span=26).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal'] = df['MACD'].ewm(span=9).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], name='MACD'))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Signal'], name='Signal'))
        recommendation = "Buy" if df['MACD'].iloc[-1] > df['Signal'].iloc[-1] else "Sell"
    
    elif analysis_type == 'Bollinger Bands':
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['StdDev'] = df['Close'].rolling(window=20).std()
        df['Upper'] = df['MA20'] + (df['StdDev'] * 2)
        df['Lower'] = df['MA20'] - (df['StdDev'] * 2)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close Price'))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Upper'], name='Upper Band'))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Lower'], name='Lower Band'))
        recommendation = "Buy" if df['Close'].iloc[-1] < df['Lower'].iloc[-1] else "Sell" if df['Close'].iloc[-1] > df['Upper'].iloc[-1] else "Hold"
    
    elif analysis_type == 'Exponential Moving Averages (EMA)':
        df['EMA10'] = df['Close'].ewm(span=10).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close Price'))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA10'], name='10-Day EMA'))
        recommendation = "Buy" if df['Close'].iloc[-1] > df['EMA10'].iloc[-1] else "Sell"
    
    return fig, recommendation

# Streamlit interface
st.title('Cryptocurrency Price Analysis and Technical Indicators')

# Dropdown menu to select cryptocurrency (universal names used)
crypto_name = st.selectbox('Select Cryptocurrency', list(cryptocurrency_map.keys()))

# Slider to select number of days within the free limit
days = st.slider('Select number of days for historical data (max 365)', min_value=1, max_value=365, value=365)

# Fetch and display candlestick chart
df = fetch_data(crypto_name, str(days))
st.write(f"Candlestick Chart for {crypto_name}")
st.plotly_chart(plot_candlestick_chart(df))

# Technical analysis type
analysis_type = st.selectbox('Select Technical Analysis', ['Moving Averages', 'RSI', 'MACD', 'Bollinger Bands', 'Exponential Moving Averages (EMA)'])

# Perform and display technical analysis
st.write(f"Technical Analysis: {analysis_type}")
fig, recommendation = perform_technical_analysis(df, analysis_type)
st.plotly_chart(fig)

# Display recommendation
st.write(f"Recommendation: {recommendation}")

#add ML models
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
