import streamlit as st
from pycoingecko import CoinGeckoAPI
import pandas as pd
import plotly.graph_objs as go

# Initialize CoinGecko API
cg = CoinGeckoAPI()

# Fetch data function using pycoingecko
def fetch_data(symbol, days='1825'):  # 5 years = 1825 days
    # Get market chart data for the selected cryptocurrency
    data = cg.get_coin_market_chart_by_id(id=symbol, vs_currency='usd', days=days)
    
    # Prepare the DataFrame
    prices = pd.DataFrame(data['prices'], columns=['Date', 'Close'])
    prices['Date'] = pd.to_datetime(prices['Date'], unit='ms')
    df = pd.DataFrame(prices)
    
    # Get additional columns like High, Low, Open (since pycoingecko doesn't provide Open/High/Low)
    df['Open'] = df['Close'].shift(1)  # Simulate 'Open' by shifting 'Close'
    df['High'] = df['Close'].rolling(window=2).max()  # Simulate 'High' by taking the rolling max
    df['Low'] = df['Close'].rolling(window=2).min()   # Simulate 'Low' by taking the rolling min
    df.dropna(inplace=True)
    
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
    df.set_index('Date', inplace=True)
    recommendation = "No recommendation available."
    
    if analysis_type == 'Moving Averages':
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['SMA100'] = df['Close'].rolling(window=100).mean()
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Actual Data'))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], mode='lines', name='SMA20'))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], mode='lines', name='SMA50'))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA100'], mode='lines', name='SMA100'))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], mode='lines', name='SMA200'))
        fig.update_layout(title='Moving Averages', xaxis_title='Date', yaxis_title='Price')
        latest_data = df.iloc[-1]
        if latest_data['SMA20'] > latest_data['SMA50'] and latest_data['SMA50'] > latest_data['SMA100']:
            recommendation = "The SMA signal indicates a **strong buy** recommendation."
        elif latest_data['SMA20'] < latest_data['SMA50'] and latest_data['SMA50'] < latest_data['SMA100']:
            recommendation = "The SMA signal indicates a **strong sell** recommendation."
        else:
            recommendation = "The SMA signal indicates a **hold** recommendation."

    elif analysis_type == 'RSI':
        delta = df['Close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'))
        fig.update_layout(title='Relative Strength Index (RSI)', xaxis_title='Date', yaxis_title='RSI')
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        latest_rsi = df['RSI'].iloc[-1]
        if latest_rsi > 70:
            recommendation = "The RSI signal indicates the asset is **overbought**. Consider selling."
        elif latest_rsi < 30:
            recommendation = "The RSI signal indicates the asset is **oversold**. Consider buying."
        else:
            recommendation = "The RSI signal indicates the asset is **neutral**. No clear action recommended."

    elif analysis_type == 'MACD':
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Signal Line'], mode='lines', name='Signal Line'))
        fig.update_layout(title='MACD', xaxis_title='Date', yaxis_title='Value')
        latest_macd = df['MACD'].iloc[-1]
        latest_signal = df['Signal Line'].iloc[-1]
        if latest_macd > latest_signal:
            recommendation = "The MACD signal indicates a **buy** recommendation."
        elif latest_macd < latest_signal:
            recommendation = "The MACD signal indicates a **sell** recommendation."
        else:
            recommendation = "The MACD signal indicates a **hold** recommendation."

    elif analysis_type == 'Bollinger Bands':
        df['MA'] = df['Close'].rolling(window=20).mean()
        df['STD'] = df['Close'].rolling(window=20).std()
        df['Upper Band'] = df['MA'] + (2 * df['STD'])
        df['Lower Band'] = df['MA'] - (2 * df['STD'])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Upper Band'], mode='lines', name='Upper Band'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Lower Band'], mode='lines', name='Lower Band'))
        fig.update_layout(title='Bollinger Bands', xaxis_title='Date', yaxis_title='Price')
        latest_close = df['Close'].iloc[-1]
        latest_upper = df['Upper Band'].iloc[-1]
        latest_lower = df['Lower Band'].iloc[-1]
        if latest_close > latest_upper:
            recommendation = "The Bollinger Bands signal indicates the asset is **overbought**."
        elif latest_close < latest_lower:
            recommendation = "The Bollinger Bands signal indicates the asset is **oversold**."
        else:
            recommendation = "The Bollinger Bands signal indicates the asset is **neutral**."

    elif analysis_type == 'Exponential Moving Averages (EMA)':
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
        df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Actual Data'))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], mode='lines', name='EMA20'))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], mode='lines', name='EMA50'))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA100'], mode='lines', name='EMA100'))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA200'], mode='lines', name='EMA200'))
        fig.update_layout(title='Exponential Moving Averages (EMA)', xaxis_title='Date', yaxis_title='Price')
        latest_data = df.iloc[-1]
        if latest_data['EMA20'] > latest_data['EMA50'] and latest_data['EMA50'] > latest_data['EMA100']:
            recommendation = "The EMA signal indicates a **strong buy** recommendation."
        elif latest_data['EMA20'] < latest_data['EMA50'] and latest_data['EMA50'] < latest_data['EMA100']:
            recommendation = "The EMA signal indicates a **strong sell** recommendation."
        else:
            recommendation = "The EMA signal indicates a **hold** recommendation."

    return fig, recommendation

# Streamlit interface
st.title('Cryptocurrency Price Analysis and Technical Indicators')

# Dropdown menu to select cryptocurrency
symbol = st.selectbox('Select Cryptocurrency', ['bitcoin', 'ethereum', 'litecoin', 'dogecoin'])

# Fetch and display candlestick chart
df = fetch_data(symbol)
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
