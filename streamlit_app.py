import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pycaret.time_series import load_model, predict_model
import os
import requests

# Function to load data
def load_data(symbols, start_date, end_date):
    dfs = {}
    for symbol in symbols:
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        dfs[symbol] = df
    return dfs

# Function to plot candlestick chart
def plot_candlestick(df, symbol):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close']
    )])
    fig.update_layout(
        title=f'Interactive Candlestick Chart for {symbol} Price',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=True
    )
    return fig

# Function for technical analysis
def technical_analysis(df, analysis_type):
    if analysis_type == 'Moving Averages':
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['MA50'] = df['close'].rolling(window=50).mean()
        df['MA100'] = df['close'].rolling(window=100).mean()
        df['MA200'] = df['close'].rolling(window=200).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Actual Data'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], mode='lines', name='MA20'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], mode='lines', name='MA50'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA100'], mode='lines', name='MA100'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA200'], mode='lines', name='MA200'))
        fig.update_layout(title='Moving Averages', xaxis_title='Date', yaxis_title='Price')
        return fig

    elif analysis_type == 'RSI':
        df['Price Change'] = df['close'].diff()
        df['Gain'] = df['Price Change'].apply(lambda x: x if x > 0 else 0)
        df['Loss'] = df['Price Change'].apply(lambda x: abs(x) if x < 0 else 0)
        window = 14
        df['Avg Gain'] = df['Gain'].rolling(window=window, min_periods=1).mean()
        df['Avg Loss'] = df['Loss'].rolling(window=window, min_periods=1).mean()
        df['RS'] = df['Avg Gain'] / df['Avg Loss']
        df['RSI'] = 100 - (100 / (1 + df['RS']))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'))
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig.update_layout(title='Relative Strength Index (RSI)', xaxis_title='Date', yaxis_title='RSI')
        return fig

    elif analysis_type == 'MACD':
        df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Signal Line'], mode='lines', name='Signal Line'))
        fig.update_layout(title='Moving Average Convergence Divergence (MACD)', xaxis_title='Date', yaxis_title='MACD')
        return fig

    elif analysis_type == 'Bollinger Bands':
        window = 20
        num_std = 2
        df['MA'] = df['close'].rolling(window=window).mean()
        df['STD'] = df['close'].rolling(window=window).std()
        df['Upper Band'] = df['MA'] + (num_std * df['STD'])
        df['Lower Band'] = df['MA'] - (num_std * df['STD'])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Close'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA'], mode='lines', name='Moving Average'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Upper Band'], mode='lines', name='Upper Bollinger Band'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Lower Band'], mode='lines', name='Lower Bollinger Band'))
        fig.update_layout(title='Bollinger Bands', xaxis_title='Date', yaxis_title='Price')
        return fig

    elif analysis_type == 'On-Balance Volume (OBV)':
        df['Price Change'] = df['close'].diff()
        df['Direction'] = df['Price Change'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        df['OBV'] = (df['Direction'] * df['volume']).cumsum()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['OBV'], mode='lines', name='OBV'))
        fig.update_layout(title='On-Balance Volume (OBV)', xaxis_title='Date', yaxis_title='OBV')
        return fig

    elif analysis_type == 'Exponential Moving Averages (EMA)':
        df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['EMA100'] = df['close'].ewm(span=100, adjust=False).mean()
        df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Actual Data'))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], mode='lines', name='EMA20'))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], mode='lines', name='EMA50'))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA100'], mode='lines', name='EMA100'))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA200'], mode='lines', name='EMA200'))
        fig.update_layout(title='Exponential Moving Averages (EMA)', xaxis_title='Date', yaxis_title='Price')
        return fig

    elif analysis_type == 'Stochastic Oscillator':
        window = 14
        smooth_window = 3
        df['Lowest Low'] = df['low'].rolling(window=window).min()
        df['Highest High'] = df['high'].rolling(window=window).max()
        df['%K'] = ((df['close'] - df['Lowest Low']) / (df['Highest High'] - df['Lowest Low'])) * 100
        df['%D'] = df['%K'].rolling(window=smooth_window).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['%K'], mode='lines', name='%K'))
        fig.add_trace(go.Scatter(x=df.index, y=df['%D'], mode='lines', name='%D'))
        fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig.update_layout(title='Stochastic Oscillator', xaxis_title='Date', yaxis_title='Value')
        return fig

    elif analysis_type == 'Average Directional Index (ADX)':
        window = 14
        df['H-L'] = df['high'] - df['low']
        df['H-PC'] = abs(df['high'] - df['close'].shift(1))
        df['L-PC'] = abs(df['low'] - df['close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=window).mean()

        df['DM+'] = df['high'].diff()
        df['DM-'] = df['low'].diff()
        df['DM+'] = df['DM+'].apply(lambda x: x if x > 0 else 0)
        df['DM-'] = df['DM-'].apply(lambda x: abs(x) if x < 0 else 0)

        df['DI+'] = (df['DM+'] / df['ATR']) * 100
        df['DI-'] = (df['DM-'] / df['ATR']) * 100
        df['DX'] = (abs(df['DI+'] - df['DI-']) / (df['DI+'] + df['DI-'])) * 100
        df['ADX'] = df['DX'].rolling(window=window).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], mode='lines', name='ADX'))
        fig.add_hline(y=25, line_dash="dash", line_color="blue", annotation_text="Trend Indicator")
        fig.update_layout(title='Average Directional Index (ADX)', xaxis_title='Date', yaxis_title='ADX')
        return fig

    else:
        return None

# Download and load models
def download_model(url, save_path):
    response = requests.get(url)
    with open(save_path, 'wb') as f:
        f.write(response.content)

# Define model URLs
model_urls = {
    'btc_model': 'https://github.com/I-r-a-j/crypto_analysis/blob/main/btc_model.pkl',
    'eth_model': 'https://github.com/I-r-a-j/crypto_analysis/blob/main/eth_model.pkl',
    'ltc_model': 'https://github.com/I-r-a-j/crypto_analysis/blob/main/ltc_model.pkl',
    'doge_model': 'https://github.com/I-r-a-j/crypto_analysis/blob/main/doge_model.pkl'
}

# Download models
for model_name, model_url in model_urls.items():
    model_path = os.path.join('/tmp', model_name + '.pkl')
    if not os.path.exists(model_path):
        download_model(model_url, model_path)

# Load models
btc_model = load_model('/tmp/btc_model')
eth_model = load_model('/tmp/eth_model')
ltc_model = load_model('/tmp/ltc_model')
doge_model = load_model('/tmp/doge_model')

# Streamlit UI
st.title('Cryptocurrency Price Analysis and Prediction')

symbols = st.multiselect('Select cryptocurrency symbols:', ['BTC-USD', 'ETH-USD', 'LTC-USD', 'DOGE-USD'])
start_date = st.date_input('Select start date:', value=datetime.now() - timedelta(days=365))
end_date = st.date_input('Select end date:', value=datetime.now())
analysis_type = st.selectbox('Select analysis type:', ['Candlestick Chart', 'Moving Averages', 'RSI', 'MACD', 'Bollinger Bands', 'On-Balance Volume (OBV)', 'Exponential Moving Averages (EMA)', 'Stochastic Oscillator', 'Average Directional Index (ADX)'])

if st.button('Load Data'):
    dfs = load_data(symbols, start_date, end_date)
    for symbol, df in dfs.items():
        st.subheader(f'{symbol} Data')
        st.write(df)

        if analysis_type == 'Candlestick Chart':
            st.plotly_chart(plot_candlestick(df, symbol))

        else:
            st.plotly_chart(technical_analysis(df, analysis_type))
