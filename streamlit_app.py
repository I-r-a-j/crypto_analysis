import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from datetime import datetime, timedelta
from tradingview_ta import TA_Handler, Interval
from pycaret.time_series import load_model

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
        df['+DM'] = (df['high'] - df['high'].shift(1)).clip(lower=0)
        df['-DM'] = (df['low'].shift(1) - df['low']).clip(lower=0)
        df['+DM'] = df['+DM'][df['+DM'] > df['-DM']]
        df['-DM'] = df['-DM'][df['-DM'] > df['+DM']]
        df['ATR'] = df['TR'].rolling(window=window).mean()
        df['+DI'] = (df['+DM'] / df['ATR']).rolling(window=window).mean() * 100
        df['-DI'] = (df['-DM'] / df['ATR']).rolling(window=window).mean() * 100
        df['DX'] = (abs(df['+DI'] - df['-DI']) / abs(df['+DI'] + df['-DI'])) * 100
        df['ADX'] = df['DX'].rolling(window=window).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['+DI'], mode='lines', name='+DI'))
        fig.add_trace(go.Scatter(x=df.index, y=df['-DI'], mode='lines', name='-DI'))
        fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], mode='lines', name='ADX'))
        fig.update_layout(title='Average Directional Index (ADX)', xaxis_title='Date', yaxis_title='Value')
        return fig

# Streamlit app
st.title('Cryptocurrency Analysis')

# Time selection
st.sidebar.subheader('Select Time Period')
end_date = datetime.now()
start_date = end_date - timedelta(days=365 * 5)

# User selection
st.sidebar.subheader('Select a Cryptocurrency')
crypto = st.sidebar.selectbox('Cryptocurrency', list(crypto_symbols.keys()))
symbol = crypto_symbols[crypto]

# Load data
if st.button('Refresh Data') or 'data' not in st.session_state:
    st.session_state.data = load_data(symbol, start_date, end_date)
data = st.session_state.data

# Date range selection
st.sidebar.subheader('Select Date Range')
date_range = st.sidebar.date_input('Select date range',
                                   value=(data.index[0].date(), data.index[-1].date()),
                                   min_value=data.index[0].date(),
                                   max_value=data.index[-1].date())
start_date, end_date = date_range
filtered_data = data.loc[start_date:end_date]

# Display data
st.write(f"Showing data for {crypto} from {start_date} to {end_date}")
st.dataframe(filtered_data)

# Input for cryptocurrency symbol
crypto = st.selectbox('Select a cryptocurrency', ['BTC', 'ETH', 'LTC', 'DOGE'])

# Mapping selected cryptocurrency to yfinance symbol
symbol_mapping = {
    'BTC': 'BTC-USD',
    'ETH': 'ETH-USD',
    'LTC': 'LTC-USD',
    'DOGE': 'DOGE-USD'
}
symbol = symbol_mapping[crypto]

# Download data
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)
dfs = load_data(symbol_mapping.values(), start_date, end_date)

# Plot candlestick chart
candlestick_fig = plot_candlestick(dfs[symbol], symbol)
st.plotly_chart(candlestick_fig)

# Select analysis type
analysis_type = st.selectbox('Select Technical Analysis Type', [
    'Moving Averages', 'RSI', 'MACD', 'Bollinger Bands', 'On-Balance Volume (OBV)',
    'Exponential Moving Averages (EMA)', 'Stochastic Oscillator', 'Average Directional Index (ADX)'
])

# Plot technical analysis
tech_analysis_fig = technical_analysis(dfs[symbol], analysis_type)
st.plotly_chart(tech_analysis_fig)

# Technical analysis summary using tradingview_ta
tv_symbol_mapping = {
    'BTC': 'BTCUSDT',
    'ETH': 'ETHUSDT',
    'LTC': 'LTCUSDT',
    'DOGE': 'DOGEUSDT'
}
tv_symbol = tv_symbol_mapping[crypto]

# Fetch technical analysis summary
try:
    ta_handler = TA_Handler(
        symbol=tv_symbol,
        screener="crypto",
        exchange="BINANCE",
        interval=Interval.INTERVAL_1_DAY
    )
    ta_analysis = ta_handler.get_analysis()
    summary = ta_analysis.summary

    st.subheader(f'Technical Analysis Summary for {crypto}')
    st.write(f"RECOMMENDATION: {summary['RECOMMENDATION']}")
    st.write(f"BUY: {summary['BUY']}")
    st.write(f"SELL: {summary['SELL']}")
    st.write(f"NEUTRAL: {summary['NEUTRAL']}")
except Exception as e:
    st.error(f"Error fetching technical analysis summary: {e}")

# Debugging: Display the current directory structure and model paths
import os
st.write("Current directory structure:")
for root, dirs, files in os.walk(".", topdown=True):
    for name in dirs:
        st.write(os.path.join(root, name))
    for name in files:
        st.write(os.path.join(root, name))

# Verify model paths
btc_model_path = '/mnt/data/btc_model.pkl'
eth_model_path = '/mnt/data/eth_model.pkl'
ltc_model_path = '/mnt/data/ltc_model.pkl'
doge_model_path = '/mnt/data/doge_model.pkl'

st.write("Verifying model paths:")
st.write(f"BTC model path exists: {os.path.exists(btc_model_path)}")
st.write(f"ETH model path exists: {os.path.exists(eth_model_path)}")
st.write(f"LTC model path exists: {os.path.exists(ltc_model_path)}")
st.write(f"DOGE model path exists: {os.path.exists(doge_model_path)}")

# Load model based on selected cryptocurrency
model_mapping = {
    'BTC': btc_model_path,
    'ETH': eth_model_path,
    'LTC': ltc_model_path,
    'DOGE': doge_model_path
}

try:
    model_path = model_mapping[crypto]
    model = load_model(model_path)
    st.write(f"{crypto} model loaded successfully.")
except Exception as e:
    st.error(f"Error loading {crypto} model: {e}")
