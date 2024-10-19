import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import requests
import pickle
from datetime import date, timedelta
from pycoingecko import CoinGeckoAPI  # Import CoinGeckoAPI for cryptocurrency data

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


# Initialize CoinGecko API client
cg = CoinGeckoAPI()

# Google Drive links for the models
MODEL_URLS = {
    'Bitcoin (BTC)': "https://drive.google.com/uc?export=download&id=1IWz0bn02AMIuK-AcuIVkyJ589bZjraHh",
    'Ethereum (ETH)': "https://drive.google.com/uc?export=download&id=1TlD8nxncxCAzWzjAJ508VksPETwdp71L",
    'Litecoin (LTC)': "https://drive.google.com/uc?export=download&id=1viq1j3as_FKR8J-ddYPpOxDsvT-_q0hA",
    'Dogecoin (DOGE)': "https://drive.google.com/uc?export=download&id=1JduM-DH42l92HvPrRrhRNXVJuYAChmKt"
}

# Function to download the model from Google Drive
@st.cache_resource  # Cache the model to avoid re-downloading
def load_model(url):
    response = requests.get(url)
    with open("crypto_model.pkl", "wb") as file:
        file.write(response.content)
    
    with open("crypto_model.pkl", "rb") as file:
        model = pickle.load(file)
    
    return model

# Constants
START = (date.today() - timedelta(days=365)).strftime("%Y-%m-%d")  # Restrict to past year
TODAY = date.today().strftime("%Y-%m-%d")
period = 5  # Predicting for the next 5 days

# Streamlit UI
st.title("Cryptocurrency Price Prediction (Next 5 Days)")

# Dropdown for selecting cryptocurrency
crypto_options = {
    'Bitcoin (BTC)': 'bitcoin',
    'Ethereum (ETH)': 'ethereum',
    'Litecoin (LTC)': 'litecoin',
    'Dogecoin (DOGE)': 'dogecoin'
}
selected_crypto = st.selectbox('Select Cryptocurrency', list(crypto_options.keys()))
selected_coin = crypto_options[selected_crypto]

# Load the model corresponding to the selected cryptocurrency
MODEL_URL = MODEL_URLS[selected_crypto]
model = load_model(MODEL_URL)

# Function to load cryptocurrency data from CoinGecko (past 365 days)
def load_data(coin):
    market_data = cg.get_coin_market_chart_by_id(id=coin, vs_currency='usd', days=365)
    prices = market_data['prices']
    
    # Convert to DataFrame
    data = pd.DataFrame(prices, columns=['timestamp', 'Close'])
    data['Date'] = pd.to_datetime(data['timestamp'], unit='ms')  # Convert from timestamp to date
    data = data[['Date', 'Close']]
    
    return data

# Load the data
data = load_data(selected_coin)

# Prepare the data for predictions
df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})

# Feature Engineering
df_train['SMA_10'] = df_train['y'].rolling(window=10).mean()  # 10-day Simple Moving Average
df_train['SMA_30'] = df_train['y'].rolling(window=30).mean()  # 30-day Simple Moving Average
df_train['EMA_10'] = df_train['y'].ewm(span=10, adjust=False).mean()  # 10-day Exponential Moving Average
df_train['EMA_30'] = df_train['y'].ewm(span=30, adjust=False).mean()  # 30-day Exponential Moving Average

df_train['day'] = df_train['ds'].dt.day
df_train['month'] = df_train['ds'].dt.month
df_train['year'] = df_train['ds'].dt.year

# Drop rows with NaN values
df_train = df_train.dropna()

# Ensure feature order matches the training data
features_order = ['SMA_10', 'SMA_30', 'EMA_10', 'EMA_30', 'day', 'month', 'year']

# Show raw data (only the last 10 rows, excluding the last row)
st.subheader(f"Raw Data for {selected_crypto}")
st.write(data.iloc[:-1].tail(10))  # Exclude the last row and show the last 10 remaining rows

# Prepare future features for prediction
today = pd.Timestamp(TODAY)

# Generate future dates starting from today (current date)
future_dates = pd.date_range(today, periods=period, freq='D').tolist()

# Use the most recent feature data for predictions (from the last row of df_train)
last_row = df_train.tail(1)

# Generate new feature data for future dates (moving averages and date features)
future_features = pd.DataFrame({
    'day': [d.day for d in future_dates],
    'month': [d.month for d in future_dates],
    'year': [d.year for d in future_dates],
    'SMA_10': last_row['SMA_10'].values[0],  # Last known SMA_10 value
    'SMA_30': last_row['SMA_30'].values[0],  # Last known SMA_30 value
    'EMA_10': last_row['EMA_10'].values[0],  # Last known EMA_10 value
    'EMA_30': last_row['EMA_30'].values[0]   # Last known EMA_30 value
})

# Ensure future features match the training feature order
future_features = future_features[features_order]

# Predict future prices using the pre-trained model
future_close = model.predict(future_features)

# Create a DataFrame for the predictions
future_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_close})
future_df.set_index('Date', inplace=True)

# Display the forecast data
st.subheader(f"Predicted Prices for {selected_crypto} for the Next {period} Days")
st.write(future_df) 
