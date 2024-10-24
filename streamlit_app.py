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

# Google Drive links for the models
MODEL_URLS = {
    'Bitcoin (BTC)': "https://drive.google.com/uc?export=download&id=15fpJ48AGZqoXSIr3kQHnCLvsg_be8D5r",
    'Ethereum (ETH)': "https://drive.google.com/uc?export=download&id=1q5I7bwqVI8_J28HXPx4DwzqHjtdTSib0",
    'Litecoin (LTC)': "https://drive.google.com/uc?export=download&id=1ay0MI8xoA2HGvhjDyRrhBoaTqSaLx7zv",
    'Dogecoin (DOGE)': "https://drive.google.com/uc?export=download&id=1ImJH3OsLPGlgDsEyg1Hllih0J-T29WuC"
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

    # Compute additional features for the model
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_30'] = df['Close'].ewm(span=30, adjust=False).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    df['day'] = df['Date'].dt.dayofweek  # Add day of the week as a feature
    
    df.dropna(inplace=True)  # Remove rows with NaN values after adding indicators
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

# Function to download the model from Google Drive
@st.cache_resource  # Cache the model to avoid re-downloading
def load_model(url):
    response = requests.get(url)
    with open("crypto_model.pkl", "wb") as file:
        file.write(response.content)
    
    with open("crypto_model.pkl", "rb") as file:
        model = pickle.load(file)
    
    return model

# Function to load cryptocurrency data from CoinGecko (past 365 days)
def load_data(coin):
    market_data = cg.get_coin_market_chart_by_id(id=coin, vs_currency='usd', days=365)
    prices = market_data['prices']
    
    # Convert to DataFrame
    data = pd.DataFrame(prices, columns=['timestamp', 'Close'])
    data['Date'] = pd.to_datetime(data['timestamp'], unit='ms')  # Convert from timestamp to date
    data = data[['Date', 'Close']]
    
    return data

# Streamlit interface
st.title('Cryptocurrency Price Prediction and Technical Analysis')
cryptocurrency = st.selectbox('Select Cryptocurrency', options=['Bitcoin (BTC)', 'Ethereum (ETH)', 'Litecoin (LTC)', 'Dogecoin (DOGE)'])

# Load the data
df = fetch_data(cryptocurrency)

# Candlestick chart display
st.subheader(f'{cryptocurrency} Candlestick Chart')
fig = plot_candlestick_chart(df)
st.plotly_chart(fig)

# Technical analysis selection
analysis_type = st.selectbox('Select Technical Analysis', ['Moving Averages', 'RSI', 'MACD', 'Bollinger Bands', 'Exponential Moving Averages (EMA)'])
fig, recommendation = perform_technical_analysis(df, analysis_type)
st.plotly_chart(fig)
st.write(f'Recommendation: {recommendation}')

# Load the pre-trained model for prediction
model_url = MODEL_URLS[cryptocurrency]
model = load_model(model_url)

# Prepare the input data for prediction
input_data = df[['Close', 'EMA_10', 'EMA_30', 'SMA_10', 'SMA_30', 'day']].iloc[-1].values.reshape(1, -1)

# Make predictions using the model
predicted_price = model.predict(input_data)
st.subheader(f'Predicted {cryptocurrency} Price for Tomorrow: ${predicted_price[0]:.2f}')
