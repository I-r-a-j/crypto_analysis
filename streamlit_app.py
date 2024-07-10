import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from tradingview_ta import TA_Handler, Interval
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
    if (analysis_type == 'Moving Averages'):
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
        df['+DM'] = df['high'].diff()
        df['-DM'] = df['low'].diff()
        df['+DM'] = df['+DM'].where((df['+DM'] > df['-DM']) & (df['+DM'] > 0), 0)
        df['-DM'] = df['-DM'].where((df['-DM'] > df['+DM']) & (df['-DM'] > 0), 0)
        df['+DI'] = 100 * (df['+DM'].ewm(alpha=1/window).mean() / df['TR'].ewm(alpha=1/window).mean())
        df['-DI'] = 100 * (df['-DM'].ewm(alpha=1/window).mean() / df['TR'].ewm(alpha=1/window).mean())
        df['DX'] = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
        df['ADX'] = df['DX'].ewm(alpha=1/window).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['+DI'], mode='lines', name='+DI'))
        fig.add_trace(go.Scatter(x=df.index, y=df['-DI'], mode='lines', name='-DI'))
        fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], mode='lines', name='ADX'))
        fig.update_layout(title='Average Directional Index (ADX)', xaxis_title='Date', yaxis_title='Value')
        return fig

    elif analysis_type == 'Ichimoku Cloud':
        # Calculation parameters
        short_span = 9
        medium_span = 26
        long_span = 52
        df['Tenkan-sen'] = (df['high'].rolling(window=short_span).max() + df['low'].rolling(window=short_span).min()) / 2
        df['Kijun-sen'] = (df['high'].rolling(window=medium_span).max() + df['low'].rolling(window=medium_span).min()) / 2
        df['Senkou Span A'] = ((df['Tenkan-sen'] + df['Kijun-sen']) / 2).shift(medium_span)
        df['Senkou Span B'] = ((df['high'].rolling(window=long_span).max() + df['low'].rolling(window=long_span).min()) / 2).shift(medium_span)
        df['Chikou Span'] = df['close'].shift(-medium_span)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Close'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Tenkan-sen'], mode='lines', name='Tenkan-sen'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Kijun-sen'], mode='lines', name='Kijun-sen'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Senkou Span A'], mode='lines', name='Senkou Span A'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Senkou Span B'], mode='lines', name='Senkou Span B', fill='tonexty'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Chikou Span'], mode='lines', name='Chikou Span'))
        fig.update_layout(title='Ichimoku Cloud', xaxis_title='Date', yaxis_title='Price')
        return fig
    elif analysis_type == 'Engulfing Pattern':
        df['Bullish Engulfing'] = (df['open'].shift(1) > df['close'].shift(1)) & (df['close'] > df['open'].shift(1)) & (df['open'] < df['close'].shift(1))
        df['Bearish Engulfing'] = (df['open'].shift(1) < df['close'].shift(1)) & (df['close'] < df['open'].shift(1)) & (df['open'] > df['close'].shift(1))
    
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'))
        fig.add_trace(go.Scatter(x=df[df['Bullish Engulfing']].index, y=df[df['Bullish Engulfing']]['low'], mode='markers', marker=dict(symbol='triangle-up', size=10, color='green'), name='Bullish Engulfing'))
        fig.add_trace(go.Scatter(x=df[df['Bearish Engulfing']].index, y=df[df['Bearish Engulfing']]['high'], mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'), name='Bearish Engulfing'))
        fig.update_layout(title='Engulfing Pattern', xaxis_title='Date', yaxis_title='Price')
        return fig
        
    elif analysis_type == 'Fibonacci Retracement':
        # Find the highest high and lowest low in the dataset
        highest_high = df['high'].max()
        lowest_low = df['low'].min()
    
        # Calculate Fibonacci levels
        diff = highest_high - lowest_low
        levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
        fib_levels = [highest_high - l * diff for l in levels]
    
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'))
    
        colors = ['purple', 'blue', 'green', 'yellow', 'orange', 'red', 'crimson']
        for fib, level, color in zip(fib_levels, levels, colors):
            fig.add_hline(y=fib, line_dash="dash", line_color=color, annotation_text=f"{level:.3f}", annotation_position="left")
    
        fig.update_layout(title='Fibonacci Retracement Levels', xaxis_title='Date', yaxis_title='Price')
        return fig
# Function to get tradingview recommendation
def get_tradingview_recommendation(tv_symbol):
    handler = TA_Handler(
        symbol=tv_symbol,
        exchange='BINANCE',
        screener='crypto',
        interval=Interval.INTERVAL_1_DAY
    )
    return handler.get_analysis().summary
# Streamlit app layout
st.title('Cryptocurrency Analysis Dashboard')
st.sidebar.title('Options')
# Define the cryptocurrency symbols
symbols = {
    'Bitcoin (BTC)': 'BTC-USD',
    'Ethereum (ETH)': 'ETH-USD',
    'Litecoin (LTC)': 'LTC-USD',
    'Dogecoin (DOGE)': 'DOGE-USD'
}
# Create a dictionary to map tradingview symbols
tv_symbols = {
    'BTC-USD': 'BTCUSDT',
    'ETH-USD': 'ETHUSDT',
    'LTC-USD': 'LTCUSDT',
    'DOGE-USD': 'DOGEUSDT'
}
# Sidebar selection for cryptocurrency
selected_crypto = st.sidebar.selectbox('Select Cryptocurrency', list(symbols.keys()))
selected_symbol = symbols[selected_crypto]
tv_symbol = tv_symbols[selected_symbol]
# Sidebar date selection
today = datetime.today()
one_year_ago = today - timedelta(days=365)
start_date = st.sidebar.date_input('Start Date', one_year_ago)
end_date = st.sidebar.date_input('End Date', today)
# Load data
dfs = load_data(symbols.values(), start_date, end_date)
# Plot candlestick chart
st.plotly_chart(plot_candlestick(dfs[selected_symbol], selected_crypto))
# Sidebar selection for technical analysis
analysis_type = st.sidebar.selectbox('Select Technical Analysis', [
    'Moving Averages', 'RSI', 'MACD', 'Bollinger Bands',
    'On-Balance Volume (OBV)', 'Exponential Moving Averages (EMA)',
    'Stochastic Oscillator', 'Average Directional Index (ADX)', 'Ichimoku Cloud' ,'Engulfing Pattern', 'Fibonacci Retracement'
])
# Perform technical analysis and plot the result
st.plotly_chart(technical_analysis(dfs[selected_symbol], analysis_type))
# TradingView technical analysis recommendations
st.subheader('TradingView Recommendations')
recommendations = get_tradingview_recommendation(tv_symbol)
st.write(recommendations)
# Moving Average Signal Recommendation
st.subheader('Moving Average Signal Recommendation')
latest_data = dfs[selected_symbol].iloc[-1]
if latest_data['MA20'] > latest_data['MA50'] and latest_data['MA50'] > latest_data['MA100']:
    st.write("The moving average signal indicates a **strong buy** recommendation.")
elif latest_data['MA20'] < latest_data['MA50'] and latest_data['MA50'] < latest_data['MA100']:
    st.write("The moving average signal indicates a **strong sell** recommendation.")
else:
    st.write("The moving average signal indicates a **hold** recommendation.")
# RSI Signal Recommendation
st.subheader('RSI Signal Recommendation')
# Ensure RSI is calculated before making a recommendation
df = dfs[selected_symbol]
df['Price Change'] = df['close'].diff()
df['Gain'] = df['Price Change'].apply(lambda x: x if x > 0 else 0)
df['Loss'] = df['Price Change'].apply(lambda x: abs(x) if x < 0 else 0)
window = 14
df['Avg Gain'] = df['Gain'].rolling(window=window, min_periods=1).mean()
df['Avg Loss'] = df['Loss'].rolling(window=window, min_periods=1).mean()
df['RS'] = df['Avg Gain'] / df['Avg Loss']
df['RSI'] = 100 - (100 / (1 + df['RS']))
# Calculate RSI and provide recommendation
latest_rsi = df['RSI'].iloc[-1]
if latest_rsi > 70:
    st.write("The RSI signal indicates the asset is **overbought**. Consider selling.")
elif latest_rsi < 30:
    st.write("The RSI signal indicates the asset is **oversold**. Consider buying.")
else:
    st.write("The RSI signal indicates the asset is **neutral**. No clear action recommended.")
# MACD Signal Recommendation
st.subheader('MACD Signal Recommendation')
# Ensure MACD is calculated before making a recommendation
df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
df['MACD'] = df['EMA12'] - df['EMA26']
df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
# Calculate MACD and provide recommendation
latest_macd = df['MACD'].iloc[-1]
latest_signal = df['Signal Line'].iloc[-1]
if latest_macd > latest_signal:
    st.write("The MACD signal indicates a **bullish** trend. Consider buying.")
elif latest_macd < latest_signal:
    st.write("The MACD signal indicates a **bearish** trend. Consider selling.")
else:
    st.write("The MACD signal indicates a **neutral** trend. No clear action recommended.")
# Bollinger Bands Signal Recommendation
st.subheader('Bollinger Bands Signal Recommendation')
# Ensure Bollinger Bands are calculated before making a recommendation
window = 20
num_std = 2
df['MA'] = df['close'].rolling(window=window).mean()
df['STD'] = df['close'].rolling(window=window).std()
df['Upper Band'] = df['MA'] + (num_std * df['STD'])
df['Lower Band'] = df['MA'] - (num_std * df['STD'])
# Calculate Bollinger Bands and provide recommendation
latest_close = df['close'].iloc[-1]
latest_upper_band = df['Upper Band'].iloc[-1]
latest_lower_band = df['Lower Band'].iloc[-1]
if latest_close > latest_upper_band:
    st.write("The Bollinger Bands signal indicates the asset is **overbought**. Consider selling.")
elif latest_close < latest_lower_band:
    st.write("The Bollinger Bands signal indicates the asset is **oversold**. Consider buying.")
else:
    st.write("The Bollinger Bands signal indicates the asset is **within normal range**. No clear action recommended.")
# OBV Signal Recommendation
st.subheader('On-Balance Volume (OBV) Signal Recommendation')
# Ensure OBV is calculated before making a recommendation
df['Price Change'] = df['close'].diff()
df['Direction'] = df['Price Change'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
df['OBV'] = (df['Direction'] * df['volume']).cumsum()
# Calculate OBV and provide recommendation
latest_obv = df['OBV'].iloc[-1]
previous_obv = df['OBV'].iloc[-2]
if latest_obv > previous_obv:
    st.write("The OBV signal indicates a **bullish** trend. Consider buying.")
elif latest_obv < previous_obv:
    st.write("The OBV signal indicates a **bearish** trend. Consider selling.")
else:
    st.write("The OBV signal indicates a **neutral** trend. No clear action recommended.")
# EMA Signal Recommendation
st.subheader('Exponential Moving Averages (EMA) Signal Recommendation')
# Ensure EMAs are calculated before making a recommendation
df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
df['EMA100'] = df['close'].ewm(span=100, adjust=False).mean()
df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()
# Calculate EMA signals and provide recommendation
latest_close = df['close'].iloc[-1]
latest_ema20 = df['EMA20'].iloc[-1]
latest_ema50 = df['EMA50'].iloc[-1]
latest_ema100 = df['EMA100'].iloc[-1]
latest_ema200 = df['EMA200'].iloc[-1]
if latest_close > latest_ema20 > latest_ema50 > latest_ema100 > latest_ema200:
    st.write("The EMA signal indicates a **strong bullish** trend. Consider buying.")
elif latest_close < latest_ema20 < latest_ema50 < latest_ema100 < latest_ema200:
    st.write("The EMA signal indicates a **strong bearish** trend. Consider selling.")
else:
    st.write("The EMA signal indicates a **neutral** trend. No clear action recommended.")
# Stochastic Oscillator Signal Recommendation
st.subheader('Stochastic Oscillator Signal Recommendation')
# Ensure Stochastic Oscillator is calculated before making a recommendation
window = 14
smooth_window = 3
df['Lowest Low'] = df['low'].rolling(window=window).min()
df['Highest High'] = df['high'].rolling(window=window).max()
df['%K'] = ((df['close'] - df['Lowest Low']) / (df['Highest High'] - df['Lowest Low'])) * 100
df['%D'] = df['%K'].rolling(window=smooth_window).mean()
# Calculate Stochastic Oscillator signals and provide recommendation
latest_k = df['%K'].iloc[-1]
latest_d = df['%D'].iloc[-1]
if latest_k > 80 and latest_d > 80:
    st.write("The Stochastic Oscillator indicates an **overbought** condition. Consider selling.")
elif latest_k < 20 and latest_d < 20:
    st.write("The Stochastic Oscillator indicates an **oversold** condition. Consider buying.")
else:
    st.write("The Stochastic Oscillator indicates a **neutral** condition. No clear action recommended.")
# ADX Signal Recommendation
st.subheader('Average Directional Index (ADX) Signal Recommendation')
# Ensure ADX is calculated before making a recommendation
window = 14
df['H-L'] = df['high'] - df['low']
df['H-PC'] = abs(df['high'] - df['close'].shift(1))
df['L-PC'] = abs(df['low'] - df['close'].shift(1))
df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
df['+DM'] = df['high'].diff()
df['-DM'] = df['low'].diff()
df['+DM'] = df['+DM'].where((df['+DM'] > df['-DM']) & (df['+DM'] > 0), 0)
df['-DM'] = df['-DM'].where((df['-DM'] > df['+DM']) & (df['-DM'] > 0), 0)
df['+DI'] = 100 * (df['+DM'].ewm(alpha=1/window).mean() / df['TR'].ewm(alpha=1/window).mean())
df['-DI'] = 100 * (df['-DM'].ewm(alpha=1/window).mean() / df['TR'].ewm(alpha=1/window).mean())
df['DX'] = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
df['ADX'] = df['DX'].ewm(alpha=1/window).mean()
# Calculate ADX signals and provide recommendation
latest_adx = df['ADX'].iloc[-1]
latest_pdi = df['+DI'].iloc[-1]
latest_ndi = df['-DI'].iloc[-1]
if latest_adx > 25 and latest_pdi > latest_ndi:
    st.write("The ADX signal indicates a **strong bullish** trend. Consider buying.")
elif latest_adx > 25 and latest_pdi < latest_ndi:
    st.write("The ADX signal indicates a **strong bearish** trend. Consider selling.")
else:
    st.write("The ADX signal indicates a **neutral** trend. No clear action recommended.")

# Ichimoku Cloud Signal Recommendation
st.subheader('Ichimoku Cloud Signal Recommendation')

# Ensure Ichimoku Cloud is calculated before making a recommendation
short_span = 9
medium_span = 26
long_span = 52
df['Tenkan-sen'] = (df['high'].rolling(window=short_span).max() + df['low'].rolling(window=short_span).min()) / 2
df['Kijun-sen'] = (df['high'].rolling(window=medium_span).max() + df['low'].rolling(window=medium_span).min()) / 2
df['Senkou Span A'] = ((df['Tenkan-sen'] + df['Kijun-sen']) / 2).shift(medium_span)
df['Senkou Span B'] = ((df['high'].rolling(window=long_span).max() + df['low'].rolling(window=long_span).min()) / 2).shift(medium_span)
df['Chikou Span'] = df['close'].shift(-medium_span)

# Calculate Ichimoku Cloud signals and provide recommendation
latest_close = df['close'].iloc[-1]
latest_tenkan_sen = df['Tenkan-sen'].iloc[-1]
latest_kijun_sen = df['Kijun-sen'].iloc[-1]
latest_senkou_span_a = df['Senkou Span A'].iloc[-1]
latest_senkou_span_b = df['Senkou Span B'].iloc[-1]
latest_chikou_span = df['Chikou Span'].iloc[-1]

if latest_close > latest_senkou_span_a and latest_close > latest_senkou_span_b:
    st.write("The Ichimoku Cloud signal indicates a **strong bullish** trend. Consider buying.")
elif latest_close < latest_senkou_span_a and latest_close < latest_senkou_span_b:
    st.write("The Ichimoku Cloud signal indicates a **strong bearish** trend. Consider selling.")
else:
    st.write("The Ichimoku Cloud signal indicates a **neutral** trend. No clear action recommended.")

# Engulfing Pattern Signal Recommendation
st.subheader('Engulfing Pattern Signal Recommendation')

# Ensure Engulfing Pattern is calculated before making a recommendation
df['Bullish Engulfing'] = (df['open'].shift(1) > df['close'].shift(1)) & (df['close'] > df['open'].shift(1)) & (df['open'] < df['close'].shift(1))
df['Bearish Engulfing'] = (df['open'].shift(1) < df['close'].shift(1)) & (df['close'] < df['open'].shift(1)) & (df['open'] > df['close'].shift(1))

# Check for recent engulfing patterns
last_5_days = df.tail(5)
bullish_engulfing = last_5_days['Bullish Engulfing'].any()
bearish_engulfing = last_5_days['Bearish Engulfing'].any()

if bullish_engulfing and not bearish_engulfing:
    st.write("A recent **Bullish Engulfing** pattern has been detected. This suggests a potential upward trend. Consider buying.")
elif bearish_engulfing and not bullish_engulfing:
    st.write("A recent **Bearish Engulfing** pattern has been detected. This suggests a potential downward trend. Consider selling.")
elif bullish_engulfing and bearish_engulfing:
    st.write("Both Bullish and Bearish Engulfing patterns have been detected recently. The market may be volatile. Exercise caution.")
else:
    st.write("No clear Engulfing patterns have been detected recently. The trend is unclear based on this indicator alone.")

# Fibonacci Retracement Signal Recommendation
st.subheader('Fibonacci Retracement Signal Recommendation')

# Calculate Fibonacci levels
highest_high = df['high'].max()
lowest_low = df['low'].min()
diff = highest_high - lowest_low
levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
fib_levels = [highest_high - l * diff for l in levels]

# Get the latest closing price
latest_close = df['close'].iloc[-1]

# Determine which Fibonacci levels the price is between
for i in range(len(fib_levels) - 1):
    if fib_levels[i+1] <= latest_close <= fib_levels[i]:
        lower_level = levels[i+1]
        upper_level = levels[i]
        break

st.write(f"The current price is between the {lower_level:.3f} and {upper_level:.3f} Fibonacci retracement levels.")

if latest_close > fib_levels[3]:  # Above 0.5 level
    st.write("The price is in the upper half of the Fibonacci range, suggesting a **bullish** trend. Consider buying or holding, but be aware of potential resistance at higher levels.")
elif latest_close < fib_levels[3]:  # Below 0.5 level
    st.write("The price is in the lower half of the Fibonacci range, suggesting a **bearish** trend. Consider selling or waiting for a bounce, but be aware of potential support at lower levels.")
else:
    st.write("The price is at the 0.5 Fibonacci level, which is often considered a pivotal point. The trend could go either way from here. Consider waiting for a clearer signal.")

st.write("Remember that Fibonacci retracement levels are best used in conjunction with other technical indicators for more reliable trading signals.")
