import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import joblib
from pycaret.time_series import load_model, predict_model

# Function to load data
import yfinance as yf

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
        df['TR14'] = df['TR'].rolling(window=window).sum()
        df['+DM14'] = df['+DM'].rolling(window=window).sum()
        df['-DM14'] = df['-DM'].rolling(window=window).sum()
        df['+DI14'] = 100 * df['+DM14'] / df['TR14']
        df['-DI14'] = 100 * df['-DM14'] / df['TR14']
        df['DX'] = 100 * abs(df['+DI14'] - df['-DI14']) / (df['+DI14'] + df['-DI14'])
        df['ADX'] = df['DX'].rolling(window=window).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], mode='lines', name='ADX'))
        fig.add_hline(y=25, line_dash="dash", line_color="green", annotation_text="Weak Trend")
        fig.add_hline(y=50, line_dash="dash", line_color="orange", annotation_text="Strong Trend")
        fig.add_hline(y=75, line_dash="dash", line_color="red", annotation_text="Very Strong Trend")
        fig.update_layout(title='Average Directional Index (ADX)', xaxis_title='Date', yaxis_title='ADX')
        return fig

    elif analysis_type == 'Ichimoku Cloud':
        df['Tenkan-sen'] = (df['high'].rolling(window=9).max() + df['low'].rolling(window=9).min()) / 2
        df['Kijun-sen'] = (df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2
        df['Senkou Span A'] = ((df['Tenkan-sen'] + df['Kijun-sen']) / 2).shift(26)
        df['Senkou Span B'] = ((df['high'].rolling(window=52).max() + df['low'].rolling(window=52).min()) / 2).shift(26)
        df['Chikou Span'] = df['close'].shift(-26)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Close'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Tenkan-sen'], mode='lines', name='Tenkan-sen'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Kijun-sen'], mode='lines', name='Kijun-sen'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Senkou Span A'], mode='lines', name='Senkou Span A'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Senkou Span B'], mode='lines', name='Senkou Span B'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Chikou Span'], mode='lines', name='Chikou Span'))
        fig.add_traces([
            go.Scatter(
                x=df.index, y=df['Senkou Span A'],
                fill=None,
                mode='lines',
                line_color='rgba(0,0,0,0)',
            ),
            go.Scatter(
                x=df.index, y=df['Senkou Span B'],
                fill='tonexty',
                mode='lines', 
                line_color='rgba(0,0,0,0)',
                fillcolor='rgba(255,200,200,0.2)',
            ),
        ])
        fig.update_layout(title='Ichimoku Cloud', xaxis_title='Date', yaxis_title='Price')
        return fig
    elif analysis_type == 'Engulfing Pattern':
        df['Body'] = df['close'] - df['open']
        engulfing_patterns = []
        for i in range(1, len(df)):
            current_candle = df.iloc[i]
            previous_candle = df.iloc[i - 1]
            if (current_candle['close'] > current_candle['open'] and
                previous_candle['close'] < previous_candle['open'] and
                current_candle['open'] < previous_candle['close'] and
                current_candle['close'] > previous_candle['open']):
                engulfing_patterns.append(current_candle.name)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Body'], mode='lines', name='Candle Body'))
        fig.add_trace(go.Scatter(x=engulfing_patterns, y=df.loc[engulfing_patterns, 'Body'], 
                                 mode='markers', name='Engulfing Pattern', 
                                 marker=dict(color='red', size=10)))
        fig.update_layout(title='Engulfing Pattern Analysis', 
                          xaxis_title='Date', 
                          yaxis_title='Body Size')
        return fig

# Load models
@st.cache(allow_output_mutation=True)
def load_ml_models():
    models = {
        'BTC-USD': load_model('models/btc_model'),
        'ETH-USD': load_model('models/eth_model'),
        'LTC-USD': load_model('models/ltc_model'),
        'DOGE-USD': load_model('models/doge_model')
    }
    return models

models = load_ml_models()

# Streamlit app
def main():
    st.title('Crypto Analysis App')

    # Section 1: Data Loading
    symbols = ['BTC-USD', 'ETH-USD', 'LTC-USD', 'DOGE-USD']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 5)

    if st.button('Refresh Data'):
        st.session_state.data = load_data(symbols, start_date, end_date)

    if 'data' not in st.session_state:
        st.session_state.data = load_data(symbols, start_date, end_date)

    # Section 2: Symbol Selection
    selected_symbol = st.selectbox('Select a cryptocurrency', symbols)
    selected_data = st.session_state.data[selected_symbol]

    # Time range selection
    date_range = st.date_input('Select date range',
                               value=(selected_data.index[0].date(), selected_data.index[-1].date()),
                               min_value=selected_data.index[0].date(),
                               max_value=selected_data.index[-1].date())

    start_date, end_date = date_range
    filtered_data = selected_data.loc[start_date:end_date]

    # Section 3: Candlestick Chart
    st.subheader('Candlestick Chart')
    candlestick_chart = plot_candlestick(filtered_data, selected_symbol)
    st.plotly_chart(candlestick_chart)

    # Section 4: Technical Analysis
    st.subheader('Technical Analysis')
    analysis_options = ['Moving Averages', 'RSI', 'MACD', 'Bollinger Bands', 'On-Balance Volume (OBV)', 
                        'Exponential Moving Averages (EMA)', 'Stochastic Oscillator', 
                        'Average Directional Index (ADX)', 'Ichimoku Cloud', 'Engulfing Pattern']
    selected_analysis = st.selectbox('Select Technical Analysis', analysis_options)

    analysis_chart = technical_analysis(filtered_data, selected_analysis)
    st.plotly_chart(analysis_chart)

    #Price predictions from pycaret models
    st.subheader('Price Prediction')
    prediction_date = st.date_input('Select date for prediction', min_value=datetime.now().date())
    
    if st.button('Predict Price'):
        # Prepare data for prediction
        last_data = filtered_data.iloc[-1].to_dict()
        last_data['Date'] = prediction_date
        predict_df = pd.DataFrame([last_data])
        
        # Make prediction
        prediction = predict_model(models[selected_symbol], data=predict_df)
        predicted_price = prediction['prediction_label'].iloc[0]
        
            st.write(f"Predicted price for {selected_symbol} on {prediction_date}: ${predicted_price:.2f}")


if __name__ == '__main__':
    main()
