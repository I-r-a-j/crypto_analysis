import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go

# Fetch data function with progress disabled
def fetch_data(symbol, period='5y'):
    df = yf.download(symbol, period=period, progress=False)
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

    elif analysis_type == 'MACD':
        df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Signal Line'], mode='lines', name='Signal Line'))
        fig.update_layout(title='Moving Average Convergence Divergence (MACD)', xaxis_title='Date', yaxis_title='MACD')

        # Generate recommendation
        latest_macd = df['MACD'].iloc[-1]
        latest_signal = df['Signal Line'].iloc[-1]
        if latest_macd > latest_signal:
            recommendation = "The MACD signal indicates a **bullish** trend. Consider buying."
        elif latest_macd < latest_signal:
            recommendation = "The MACD signal indicates a **bearish** trend. Consider selling."
        else:
            recommendation = "The MACD signal indicates a **neutral** trend. No clear action recommended."

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

        # Generate recommendation
        latest_close = df['close'].iloc[-1]
        latest_upper_band = df['Upper Band'].iloc[-1]
        latest_lower_band = df['Lower Band'].iloc[-1]
        if latest_close > latest_upper_band:
            recommendation = "The Bollinger Bands signal indicates the asset is **overbought**. Consider selling."
        elif latest_close < latest_lower_band:
            recommendation = "The Bollinger Bands signal indicates the asset is **oversold**. Consider buying."
        else:
            recommendation = "The Bollinger Bands signal indicates the asset is **within normal range**. No clear action recommended."

    elif analysis_type == 'On-Balance Volume (OBV)':
        df['Price Change'] = df['close'].diff()
        df['Direction'] = df['Price Change'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        df['OBV'] = (df['Direction'] * df['volume']).cumsum()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['OBV'], mode='lines', name='OBV'))
        fig.update_layout(title='On-Balance Volume (OBV)', xaxis_title='Date', yaxis_title='OBV')

        # Generate recommendation
        latest_obv = df['OBV'].iloc[-1]
        previous_obv = df['OBV'].iloc[-2]
        if latest_obv > previous_obv:
            recommendation = "The OBV signal indicates a **bullish** trend. Consider buying."
        elif latest_obv < previous_obv:
            recommendation = "The OBV signal indicates a **bearish** trend. Consider selling."
        else:
            recommendation = "The OBV signal indicates a **neutral** trend. No clear action recommended."

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

        # Generate recommendation
        latest_data = df.iloc[-1]
        if latest_data['EMA20'] > latest_data['EMA50'] and latest_data['EMA50'] > latest_data['EMA100']:
            recommendation = "The EMA signal indicates a **strong buy** recommendation."
        elif latest_data['EMA20'] < latest_data['EMA50'] and latest_data['EMA50'] < latest_data['EMA100']:
            recommendation = "The EMA signal indicates a **strong sell** recommendation."
        else:
            recommendation = "The EMA signal indicates a **hold** recommendation."

    elif analysis_type == 'Stochastic Oscillator':
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        df['%K'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
        df['%D'] = df['%K'].rolling(window=3).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['%K'], mode='lines', name='%K'))
        fig.add_trace(go.Scatter(x=df.index, y=df['%D'], mode='lines', name='%D'))
        fig.update_layout(title='Stochastic Oscillator', xaxis_title='Date', yaxis_title='Value')
        fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="Oversold")

        # Generate recommendation
        latest_k = df['%K'].iloc[-1]
        if latest_k > 80:
            recommendation = "The Stochastic Oscillator signal indicates the asset is **overbought**. Consider selling."
        elif latest_k < 20:
            recommendation = "The Stochastic Oscillator signal indicates the asset is **oversold**. Consider buying."
        else:
            recommendation = "The Stochastic Oscillator signal indicates the asset is **neutral**. No clear action recommended."

    elif analysis_type == 'Average Directional Index (ADX)':
        df['TR'] = df[['high', 'low', 'close']].diff().abs().max(axis=1)
        df['+DM'] = df['high'].diff().apply(lambda x: x if x > 0 else 0)
        df['-DM'] = df['low'].diff().apply(lambda x: abs(x) if x < 0 else 0)
        df['+DI'] = 100 * (df['+DM'].ewm(span=14).mean() / df['TR'].ewm(span=14).mean())
        df['-DI'] = 100 * (df['-DM'].ewm(span=14).mean() / df['TR'].ewm(span=14).mean())
        df['DX'] = 100 * (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI']))
        df['ADX'] = df['DX'].ewm(span=14).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], mode='lines', name='ADX'))
        fig.update_layout(title='Average Directional Index (ADX)', xaxis_title='Date', yaxis_title='ADX')

        # Generate recommendation
        latest_adx = df['ADX'].iloc[-1]
        if latest_adx > 25:
            recommendation = "The ADX signal indicates a **strong trend**. Consider trading."
        else:
            recommendation = "The ADX signal indicates a **weak trend**. Consider holding."

    elif analysis_type == 'Ichimoku Cloud':
        df['tenkan_sen'] = (df['high'].rolling(window=9).max() + df['low'].rolling(window=9).min()) / 2
        df['kijun_sen'] = (df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
        df['senkou_span_b'] = ((df['high'].rolling(window=52).max() + df['low'].rolling(window=52).min()) / 2).shift(26)
        df['chikou_span'] = df['close'].shift(-26)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['tenkan_sen'], mode='lines', name='Tenkan-sen'))
        fig.add_trace(go.Scatter(x=df.index, y=df['kijun_sen'], mode='lines', name='Kijun-sen'))
        fig.add_trace(go.Scatter(x=df.index, y=df['senkou_span_a'], mode='lines', name='Senkou Span A'))
        fig.add_trace(go.Scatter(x=df.index, y=df['senkou_span_b'], mode='lines', name='Senkou Span B'))
        fig.add_trace(go.Scatter(x=df.index, y=df['chikou_span'], mode='lines', name='Chikou Span'))
        fig.update_layout(title='Ichimoku Cloud', xaxis_title='Date', yaxis_title='Value')

        # Generate recommendation
        latest_close = df['close'].iloc[-1]
        latest_senkou_a = df['senkou_span_a'].iloc[-1]
        latest_senkou_b = df['senkou_span_b'].iloc[-1]
        if latest_close > latest_senkou_a and latest_close > latest_senkou_b:
            recommendation = "The Ichimoku Cloud signal indicates a **bullish** trend. Consider buying."
        elif latest_close < latest_senkou_a and latest_close < latest_senkou_b:
            recommendation = "The Ichimoku Cloud signal indicates a **bearish** trend. Consider selling."
        else:
            recommendation = "The Ichimoku Cloud signal indicates a **neutral** trend. No clear action recommended."

    elif analysis_type == 'Engulfing Pattern':
        df['Engulfing'] = ((df['open'].shift(1) > df['close'].shift(1)) & (df['open'] < df['close']) & (df['open'] < df['close'].shift(1)) & (df['close'] > df['open'].shift(1))) | \
                          ((df['open'].shift(1) < df['close'].shift(1)) & (df['open'] > df['close']) & (df['open'] > df['close'].shift(1)) & (df['close'] < df['open'].shift(1)))

        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close']
        )])
        engulfing_df = df[df['Engulfing']]
        fig.add_trace(go.Scatter(
            x=engulfing_df.index,
            y=engulfing_df['close'],
            mode='markers',
            marker=dict(size=10, color='red'),
            name='Engulfing Pattern'
        ))
        fig.update_layout(title='Engulfing Pattern', xaxis_title='Date', yaxis_title='Price')

        # Generate recommendation
        if not engulfing_df.empty:
            last_engulfing = engulfing_df.index[-1]
            if df.loc[last_engulfing]['close'] > df.loc[last_engulfing]['open']:
                recommendation = "The Engulfing Pattern signal indicates a **bullish** trend. Consider buying."
            else:
                recommendation = "The Engulfing Pattern signal indicates a **bearish** trend. Consider selling."
        else:
            recommendation = "No recent engulfing pattern detected. No clear action recommended."

    return fig, recommendation

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
