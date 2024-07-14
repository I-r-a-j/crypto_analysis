import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go

# Fetch data function
def fetch_data(symbol, period='5y'):
    df = yf.download(symbol, period=period)
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

    # Add similar blocks for other technical analysis types...

    return fig, recommendation

# Streamlit app
st.title("Cryptocurrency Analysis Dashboard")

# Sidebar options
st.sidebar.title("Options")
symbols = ['btc-usd', 'eth-usd', 'ltc-usd', 'doge-usd']
selected_symbol = st.sidebar.selectbox('Select Cryptocurrency', symbols)
technical_analysis_type = st.sidebar.selectbox('Select Technical Analysis Type', 
                                               ['Moving Averages', 'RSI', 'MACD', 'Bollinger Bands', 
                                                'On-Balance Volume (OBV)', 'Exponential Moving Averages (EMA)', 
                                                'Stochastic Oscillator', 'Average Directional Index (ADX)', 
                                                'Ichimoku Cloud', 'Engulfing Pattern'])

# Fetch data for selected symbol
data = fetch_data(selected_symbol)

# Display candlestick chart
st.subheader(f"{selected_symbol.upper()} Candlestick Chart")
st.plotly_chart(plot_candlestick_chart(data))

# Perform selected technical analysis
st.subheader(f"{selected_symbol.upper()} {technical_analysis_type} Analysis")
analysis_fig, recommendation = perform_technical_analysis(data, technical_analysis_type)
st.plotly_chart(analysis_fig)
st.markdown(recommendation)
