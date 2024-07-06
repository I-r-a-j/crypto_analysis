import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
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
        df['+DM'] = df['high'].diff().where(df['high'].diff() > df['low'].diff(), 0).fillna(0)
        df['-DM'] = df['low'].diff().where(df['low'].diff() > df['high'].diff(), 0).fillna(0)
        df['+DI'] = 100 * (df['+DM'].rolling(window=window).mean() / df['TR'].rolling(window=window).mean())
        df['-DI'] = 100 * (df['-DM'].rolling(window=window).mean() / df['TR'].rolling(window=window).mean())
        df['DX'] = 100 * (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI']))
        df['ADX'] = df['DX'].rolling(window=window).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], mode='lines', name='ADX'))
        fig.update_layout(title='Average Directional Index (ADX)', xaxis_title='Date', yaxis_title='ADX')
        return fig
    elif analysis_type == 'Volume-Weighted Average Price (VWAP)':
        df['TP'] = (df['high'] + df['low'] + df['close']) / 3
        df['TPV'] = df['TP'] * df['volume']
        df['Cumulative TPV'] = df['TPV'].cumsum()
        df['Cumulative Volume'] = df['volume'].cumsum()
        df['VWAP'] = df['Cumulative TPV'] / df['Cumulative Volume']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Close'))
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], mode='lines', name='VWAP'))
        fig.update_layout(title='Volume-Weighted Average Price (VWAP)', xaxis_title='Date', yaxis_title='Price')
        return fig
    else:
        return None

# Load models
btc_model = load_model('https://github.com/I-r-a-j/crypto_analysis/blob/main/btc_model.pkl')
eth_model = load_model('https://github.com/I-r-a-j/crypto_analysis/blob/main/eth_model.pkl')
ltc_model = load_model('https://github.com/I-r-a-j/crypto_analysis/blob/main/ltc_model.pkl')
doge_model = load_model('https://github.com/I-r-a-j/crypto_analysis/blob/main/doge_model.pkl')

# Define the main function
def main():
    st.title("Cryptocurrency Dashboard")

    symbols = ['BTC-USD', 'ETH-USD', 'LTC-USD', 'DOGE-USD']

    symbol_map = {
        'BTC-USD': 'BTCUSDT',
        'ETH-USD': 'ETHUSDT',
        'LTC-USD': 'LTCUSDT',
        'DOGE-USD': 'DOGEUSDT'
    }

    # Sidebar options
    symbol = st.sidebar.selectbox('Select Cryptocurrency', symbols)
    start_date = st.sidebar.date_input('Start Date', datetime.today() - timedelta(days=365*5))
    end_date = st.sidebar.date_input('End Date', datetime.today())
    analysis_type = st.sidebar.selectbox('Select Technical Analysis', 
                                         ['Moving Averages', 'RSI', 'MACD', 'Bollinger Bands', 'On-Balance Volume (OBV)', 
                                          'Exponential Moving Averages (EMA)', 'Stochastic Oscillator', 'Average Directional Index (ADX)', 
                                          'Volume-Weighted Average Price (VWAP)'])

    if 'data' not in st.session_state:
        st.session_state.data = {}

    # Load data
    st.session_state.data = load_data([symbol], start_date, end_date)

    # Extract the selected DataFrame from the dictionary
    df = st.session_state.data[symbol]

    # Display candlestick chart
    st.plotly_chart(plot_candlestick(df, symbol), use_container_width=True)

    # Display technical analysis chart
    st.plotly_chart(technical_analysis(df, analysis_type), use_container_width=True)

    # TradingView technical analysis
    tv_symbol = symbol_map[symbol]
    handler = TA_Handler(symbol=tv_symbol, screener='crypto', exchange='binance', interval=Interval.INTERVAL_1_DAY)
    analysis = handler.get_analysis()
    st.write(f"TradingView Analysis for {symbol}")
    st.write(f"Summary: {analysis.summary}")
    st.write(f"Oscillators: {analysis.oscillators['COMPUTE']}")
    st.write(f"Moving Averages: {analysis.moving_averages['COMPUTE']}")

if __name__ == '__main__':
    main()
