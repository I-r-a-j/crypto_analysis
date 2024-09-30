import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import pycaret.regression as pycr
import os

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
    elif analysis_type == 'Stochastic Oscillator':
        df['L14'] = df['Low'].rolling(window=14).min()
        df['H14'] = df['High'].rolling(window=14).max()
        df['%K'] = 100 * ((df['Close'] - df['L14']) / (df['H14'] - df['L14']))
        df['%D'] = df['%K'].rolling(window=3).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['%K'], mode='lines', name='%K'))
        fig.add_trace(go.Scatter(x=df.index, y=df['%D'], mode='lines', name='%D'))
        fig.update_layout(title='Stochastic Oscillator', xaxis_title='Date', yaxis_title='Value')
        fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="Oversold")
        latest_k = df['%K'].iloc[-1]
        latest_d = df['%D'].iloc[-1]
        if latest_k > 80:
            recommendation = "The Stochastic Oscillator indicates the asset is **overbought**. Consider selling."
        elif latest_k < 20:
            recommendation = "The Stochastic Oscillator indicates the asset is **oversold**. Consider buying."
        else:
            recommendation = "The Stochastic Oscillator indicates the asset is **neutral**. No clear action recommended."
   
    elif analysis_type == 'Ichimoku Cloud':
        df['Tenkan-sen'] = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2
        df['Kijun-sen'] = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2
        df['Senkou Span A'] = ((df['Tenkan-sen'] + df['Kijun-sen']) / 2).shift(26)
        df['Senkou Span B'] = ((df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2).shift(26)
        df['Chikou Span'] = df['Close'].shift(-26)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Tenkan-sen'], mode='lines', name='Tenkan-sen'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Kijun-sen'], mode='lines', name='Kijun-sen'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Senkou Span A'], mode='lines', fill='tonexty', name='Senkou Span A'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Senkou Span B'], mode='lines', fill='tonexty', name='Senkou Span B'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Chikou Span'], mode='lines', name='Chikou Span'))
        fig.update_layout(title='Ichimoku Cloud', xaxis_title='Date', yaxis_title='Price')
        latest_close = df['Close'].iloc[-1]
        latest_span_a = df['Senkou Span A'].iloc[-1]
        latest_span_b = df['Senkou Span B'].iloc[-1]
        if latest_close > latest_span_a and latest_close > latest_span_b:
            recommendation = "The Ichimoku Cloud indicates a **strong buy** recommendation."
        elif latest_close < latest_span_a and latest_close < latest_span_b:
            recommendation = "The Ichimoku Cloud indicates a **strong sell** recommendation."
        else:
            recommendation = "The Ichimoku Cloud indicates a **neutral** recommendation."

    return fig, recommendation

# Streamlit app
st.title("Cryptocurrency Analysis Dashboard")
# Sidebar options
st.sidebar.title("Options")
symbols = ['btc-usd', 'eth-usd', 'ltc-usd', 'doge-usd']
selected_symbol = st.sidebar.selectbox('Select Cryptocurrency', symbols)
technical_analysis_type = st.sidebar.selectbox('Select Technical Analysis Type', 
                                               ['Moving Averages', 'RSI', 'MACD', 'Bollinger Bands', 
                                                'Exponential Moving Averages (EMA)', 
                                                'Stochastic Oscillator', 
                                                'Ichimoku Cloud'])

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


# Load machine learning model
model_path = 'btc_simple_model.pkl'
if os.path.exists(model_path):
    model = pycr.load_model(model_path)
else:
    st.error("Model not found. Please check the path or the GitHub repo.")

# Make predictions using the model
def predict_with_model(df):
    prediction_df = df[['Open', 'High', 'Low', 'Close']]  # Adjust as per model features
    predictions = pycr.predict_model(model, data=prediction_df)
    return predictions['Label']

# Add prediction results to the app
if selected_symbol == 'btc-usd':
    df = fetch_data(selected_symbol)
    predictions = predict_with_model(df)
    df['Predictions'] = predictions

    st.subheader("Predicted Close Prices")
    st.line_chart(df[['Date', 'Close', 'Predictions']].set_index('Date'))
