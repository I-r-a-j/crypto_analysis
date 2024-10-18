**Disclaimer**

this porject is for demonstration only do not use it for actual trading.

**Overview**

Welcome to the Cryptocurrency Analysis Dashboard, a powerful Streamlit-based app for performing detailed technical analysis and price predictions for popular cryptocurrencies. This app allows users to view candlestick charts, apply technical indicators, and predict future prices for cryptocurrencies such as Bitcoin (BTC), Ethereum (ETH), Litecoin (LTC), and Dogecoin (DOGE).

You can explore the live app here: Cryptocurrency Analysis Dashboard

**Features**

Candlestick Chart: Visualize the price movements of selected cryptocurrencies with interactive candlestick charts.
Technical Indicators: Apply popular technical analysis tools such as:
Moving Averages (SMA)
Relative Strength Index (RSI)
Moving Average Convergence Divergence (MACD)
Bollinger Bands
Exponential Moving Averages (EMA)
Stochastic Oscillator
Ichimoku Cloud
Cryptocurrency Price Prediction: Predict the next 5 days of price movements using pre-trained machine learning models for Bitcoin (BTC), Ethereum (ETH), Litecoin (LTC), and Dogecoin (DOGE).

**Installation**

Prerequisites
Ensure you have Python installed on your system. This app requires the following Python libraries:

Streamlit
yfinance
pandas
plotly
requests
pickle
pycoingecko

*Clone the Repository*

bash
Copy code
git clone https://github.com/your-username/crypto-analysis.git
cd crypto-analysis

*Install Dependencies*

bash
Copy code
pip install -r requirements.txt

*Run the App*

bash
Copy code
streamlit run app.py

**Usage**

Launch the App: After running the app, open your browser and go to localhost:8501.
Select a Cryptocurrency: Choose a cryptocurrency from the sidebar (Bitcoin, Ethereum, Litecoin, Dogecoin).
Technical Analysis: Select one of the available technical analysis tools to view its chart and get recommendations (buy, sell, hold).
Price Prediction: In the price prediction section, select the cryptocurrency to forecast the next 5 days' prices. The app loads the relevant pre-trained model and displays the forecast.

**Model Information**

The machine learning models for predicting cryptocurrency prices are trained on historical data using technical indicators such as moving averages. Each cryptocurrency model can predict
the next 5 days of prices based on recent data.

**Contributing**

Feel free to contribute by opening issues or submitting pull requests. For major changes, please open an issue first to discuss what you would like to change.

**License**

This project is licensed under the MIT License.






