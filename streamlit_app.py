import streamlit as st
from pycoingecko import CoinGeckoAPI
import pandas as pd
import plotly.graph_objs as go

# Initialize CoinGecko API
cg = CoinGeckoAPI()

# Function to fetch cryptocurrency data
def fetch_data(symbol, days='365'):  # Limited to 365 days for free API users
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

# Dropdown menu to select cryptocurrency
symbol = st.selectbox('Select Cryptocurrency', ['bitcoin', 'ethereum', 'litecoin', 'dogecoin'])

# Slider to select number of days within the free limit
days = st.slider('Select number of days for historical data (max 365)', min_value=1, max_value=365, value=365)

# Fetch and display candlestick chart
df = fetch_data(symbol, str(days))
st.write("Candlestick Chart")
st.plotly_chart(plot_candlestick_chart(df))

# Technical analysis type
analysis_type = st.selectbox('Select Technical Analysis', ['Moving Averages', 'RSI', 'MACD', 'Bollinger Bands', 'Exponential Moving Averages (EMA)'])

# Perform and display technical analysis
st.write(f"Technical Analysis: {analysis_type}")
fig, recommendation = perform_technical_analysis(df, analysis_type)
st.plotly_chart(fig)

# Display recommendation
st.write(f"Recommendation: {recommendation}")
