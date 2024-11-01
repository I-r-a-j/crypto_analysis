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
PREDICTION_PERIOD = 5

# Configuration dictionaries
MODEL_URLS = {
    'Bitcoin (BTC)': "https://drive.google.com/uc?export=download&id=1Zo_vluWUB1D9xvo4F5yFhe1DEXr0un3V",
    'Ethereum (ETH)': "https://drive.google.com/uc?export=download&id=165tsktsGtYBWDAd7gam6H4X-eCV8inp5",
    'Litecoin (LTC)': "https://drive.google.com/uc?export=download&id=1tCwrwWp1eio4wfeoDCRcrTY_b_lW2SDu",
    'Dogecoin (DOGE)': "https://drive.google.com/uc?export=download&id=1wAcg96im3W1AzJ53lNuFs7TJaIm7vJQo"
}

CRYPTO_OPTIONS = {
    'Bitcoin (BTC)': 'bitcoin',
    'Ethereum (ETH)': 'ethereum',
    'Litecoin (LTC)': 'litecoin',
    'Dogecoin (DOGE)': 'dogecoin'
}

def load_data(coin):
    """Fetch historical price data and additional market data from CoinGecko"""
    # Get price data
    market_data = cg.get_coin_market_chart_by_id(id=coin, vs_currency='usd', days=365)
    prices = market_data['prices']
    volumes = market_data['total_volumes']
    market_caps = market_data['market_caps']
    
    # Create main DataFrame
    df = pd.DataFrame(prices, columns=['timestamp', 'Close'])
    df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Add volume and market cap
    df['Volume'] = [v[1] for v in volumes]
    df['Market_Cap'] = [m[1] for m in market_caps]
    
    # Calculate daily price changes
    df['Price_Change'] = df['Close'].pct_change() * 100
    
    # Get additional coin info
    coin_info = cg.get_coin_by_id(coin)
    
    return df, coin_info

def get_detailed_market_info(coin_info):
    """Extract relevant market information from coin info"""
    return {
        'Current Price (USD)': coin_info['market_data']['current_price']['usd'],
        '24h Change (%)': coin_info['market_data']['price_change_percentage_24h'],
        '7d Change (%)': coin_info['market_data']['price_change_percentage_7d'],
        '30d Change (%)': coin_info['market_data']['price_change_percentage_30d'],
        'Market Cap (USD)': coin_info['market_data']['market_cap']['usd'],
        'Market Cap Rank': coin_info['market_cap_rank'],
        '24h Volume': coin_info['market_data']['total_volume']['usd'],
        'Circulating Supply': coin_info['market_data']['circulating_supply'],
        'Total Supply': coin_info['market_data']['total_supply'],
        'ATH (USD)': coin_info['market_data']['ath']['usd'],
        'ATH Change (%)': coin_info['market_data']['ath_change_percentage']['usd']
    }

def create_candlestick_chart(data, title):
    """Create a candlestick chart"""
    fig = go.Figure(data=[go.Candlestick(
        x=data['Date'],
        open=data['Close'].shift(1),
        high=data['Close'] * 1.001,  # Simulating high price
        low=data['Close'] * 0.999,   # Simulating low price
        close=data['Close'],
        name='Price'
    )])
    
    fig.update_layout(
        title=title,
        yaxis_title='Price (USD)',
        xaxis_title='Date',
        template='plotly_dark',
        height=600
    )
    
    return fig

def create_technical_chart(data, indicator, params=None):
    """Create a technical analysis chart"""
    fig = go.Figure()
    
    # Add price line only for Moving Average and Bollinger Bands
    if indicator in ['Moving Average', 'Bollinger Bands']:
        fig.add_trace(go.Scatter(
            x=data['Date'],
            y=data['Close'],
            name='Price',
            line=dict(color='lightgray')
        ))
    
    # Add technical indicator
    if indicator == 'Moving Average':
        window = params.get('window', 20)
        ma = data['Close'].rolling(window=window).mean()
        fig.add_trace(go.Scatter(x=data['Date'], y=ma, name=f'MA {window}'))
        
    elif indicator == 'RSI':
        window = params.get('window', 14)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        fig.add_trace(go.Scatter(x=data['Date'], y=rsi, name=f'RSI {window}'))
        
    elif indicator == 'Bollinger Bands':
        window = params.get('window', 20)
        ma = data['Close'].rolling(window=window).mean()
        std = data['Close'].rolling(window=window).std()
        fig.add_trace(go.Scatter(x=data['Date'], y=ma + 2*std, name='Upper Band'))
        fig.add_trace(go.Scatter(x=data['Date'], y=ma - 2*std, name='Lower Band'))
        
    elif indicator == 'MACD':
        short_ema = data['Close'].ewm(span=12, adjust=False).mean()
        long_ema = data['Close'].ewm(span=26, adjust=False).mean()
        macd = short_ema - long_ema
        signal = macd.ewm(span=9, adjust=False).mean()
        fig.add_trace(go.Scatter(x=data['Date'], y=macd, name='MACD'))
        fig.add_trace(go.Scatter(x=data['Date'], y=signal, name='Signal'))
    
    fig.update_layout(
        title=f'{indicator} Analysis',
        yaxis_title='Value',
        xaxis_title='Date',
        template='plotly_dark',
        height=400
    )
    
    return fig
    
def main():
    st.title("Cryptocurrency Analysis Dashboard")
    
    # Sidebar for cryptocurrency selection
    selected_crypto = st.sidebar.selectbox('Select Cryptocurrency', list(CRYPTO_OPTIONS.keys()))
    selected_coin = CRYPTO_OPTIONS[selected_crypto]
    
    # Load data
    data, coin_info = load_data(selected_coin)
    
    # Market Information Section
    st.header("Market Information")
    market_info = get_detailed_market_info(coin_info)
    
    # Create three columns for market info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Price", f"${market_info['Current Price (USD)']:,.2f}", 
                 f"{market_info['24h Change (%)']:.2f}%")
    with col2:
        st.metric("Market Cap", f"${market_info['Market Cap (USD)']:,.0f}", 
                 f"Rank: {market_info['Market Cap Rank']}")
    with col3:
        st.metric("24h Volume", f"${market_info['24h Volume']:,.0f}")
    
    # Recent Price Data Table
    st.subheader(f"Last 5 Days Data for {selected_crypto}")
    recent_data = data.tail(5)[['Date', 'Close', 'Volume', 'Price_Change', 'Market_Cap']].copy()
    recent_data['Date'] = recent_data['Date'].dt.date
    recent_data.columns = ['Date', 'Close Price (USD)', '24h Volume', 'Daily Change (%)', 'Market Cap (USD)']
    recent_data = recent_data.set_index('Date')
    st.dataframe(recent_data.style.format({
        'Close Price (USD)': '${:,.2f}',
        'Daily Change (%)': '{:,.2f}%',
        '24h Volume': '${:,.0f}',
        'Market Cap (USD)': '${:,.0f}'
    }))
    
    # Candlestick Chart Section
    st.header("Price Chart")
    candlestick_fig = create_candlestick_chart(data, f"{selected_crypto} Price Chart")
    st.plotly_chart(candlestick_fig, use_container_width=True)
    
    # Technical Analysis Section
    st.header("Technical Analysis")
    indicator = st.selectbox(
        'Select Technical Indicator',
        ['Moving Average', 'RSI', 'MACD', 'Bollinger Bands']
    )
    
    # Parameter selection based on indicator
    params = {}
    if indicator in ['Moving Average', 'RSI', 'Bollinger Bands']:
        window = st.slider(f'Select {indicator} Window', 5, 50, 20)
        params['window'] = window
    
    # Create and display technical chart
    technical_fig = create_technical_chart(data, indicator, params)
    st.plotly_chart(technical_fig, use_container_width=True)

# Add these functions back for predictions
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
    st.title("Cryptocurrency Analysis Dashboard")
    
    # Sidebar for cryptocurrency selection
    selected_crypto = st.sidebar.selectbox('Select Cryptocurrency', list(CRYPTO_OPTIONS.keys()))
    selected_coin = CRYPTO_OPTIONS[selected_crypto]
    
    # Load data
    data, coin_info = load_data(selected_coin)
    
    # Market Information Section
    st.header("Market Information")
    market_info = get_detailed_market_info(coin_info)
    
    # [Previous market info columns code remains the same...]
    
    # Recent Price Data Table
    st.subheader(f"Last 5 Days Data for {selected_crypto}")
    recent_data = data.tail(5)[['Date', 'Close', 'Volume', 'Price_Change', 'Market_Cap']].copy()
    recent_data['Date'] = recent_data['Date'].dt.date
    recent_data.columns = ['Date', 'Close Price (USD)', '24h Volume', 'Daily Change (%)', 'Market Cap (USD)']
    recent_data = recent_data.set_index('Date')
    st.dataframe(recent_data.style.format({
        'Close Price (USD)': '${:,.2f}',
        'Daily Change (%)': '{:,.2f}%',
        '24h Volume': '${:,.0f}',
        'Market Cap (USD)': '${:,.0f}'
    }))
    
    # Price Prediction Section
    st.subheader(f"Price Predictions for Next {PREDICTION_PERIOD} Days")
    try:
        # Load and prepare model
        with st.spinner('Loading prediction model...'):
            model = load_model(MODEL_URLS[selected_crypto])
        
        # Prepare features for prediction
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
        
        # Create and display prediction DataFrame
        future_df = pd.DataFrame({
            'Date': future_dates.date,
            'Predicted Price (USD)': future_close,
        })
        future_df = future_df.set_index('Date')
        
        # Calculate daily change percentages for predictions
        current_price = data['Close'].iloc[-1]
        future_df['Predicted Change (%)'] = 0.0
        future_df.iloc[0, future_df.columns.get_loc('Predicted Change (%)')] = (
            (future_df['Predicted Price (USD)'].iloc[0] - current_price) / current_price * 100
        )
        for i in range(1, len(future_df)):
            future_df.iloc[i, future_df.columns.get_loc('Predicted Change (%)')] = (
                (future_df['Predicted Price (USD)'].iloc[i] - future_df['Predicted Price (USD)'].iloc[i-1]) 
                / future_df['Predicted Price (USD)'].iloc[i-1] * 100
            )
        
        # Display prediction table with formatting
        st.dataframe(future_df.style.format({
            'Predicted Price (USD)': '${:,.2f}',
            'Predicted Change (%)': '{:+.2f}%'
        }))
        
        # Add a note about predictions
        st.caption("""
            Note: Predictions are based on historical data and technical indicators. 
            Market conditions can change rapidly, and actual prices may vary significantly from predictions.
            """)
            
    except Exception as e:
        st.error(f"Error loading prediction model: {str(e)}")
        st.warning("Price predictions are temporarily unavailable. Please try again later.")
    
    # [Rest of the visualization code remains the same...]
    
    # Candlestick Chart Section
    st.header("Price Chart")
    candlestick_fig = create_candlestick_chart(data, f"{selected_crypto} Price Chart")
    st.plotly_chart(candlestick_fig, use_container_width=True)
    
    # Technical Analysis Section
    st.header("Technical Analysis")
    indicator = st.selectbox(
        'Select Technical Indicator',
        ['Moving Average', 'RSI', 'MACD', 'Bollinger Bands']
    )
    
    # Parameter selection based on indicator
    params = {}
    if indicator in ['Moving Average', 'RSI', 'Bollinger Bands']:
        window = st.slider(f'Select {indicator} Window', 5, 50, 20)
        params['window'] = window
    
    # Create and display technical chart
    technical_fig = create_technical_chart(data, indicator, params)
    st.plotly_chart(technical_fig, use_container_width=True)

if __name__ == "__main__":
    main()
