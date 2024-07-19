***Cryptocurrency Analysis Dashboard***

This repository hosts a Streamlit-based web application that performs technical analysis on selected cryptocurrencies using various indicators. The app fetches data from Yahoo Finance and visualizes it through interactive candlestick charts and technical analysis plots using Plotly.
you can use it in streamlit cloud with bellow link:
https://crypto-analysis.streamlit.app/

**Features**

Fetch Cryptocurrency Data: Retrieve historical data for popular cryptocurrencies (e.g., BTC, ETH, LTC, DOGE) over a specified period.
Candlestick Charts: Visualize price movements with candlestick charts.
Technical Analysis: Perform technical analysis using a range of indicators, including:
Moving Averages (SMA, EMA)
Relative Strength Index (RSI)
Moving Average Convergence Divergence (MACD)
Bollinger Bands
Stochastic Oscillator
Ichimoku Cloud

**How It Works**

1.Fetch Data Function:
Retrieves historical price data from Yahoo Finance.
Resets the index for easy manipulation.

2.Plot Candlestick Chart Function:
Creates a candlestick chart using Plotly.
Includes date and price axes for clarity.

3.Perform Technical Analysis Function:
Conducts analysis based on the selected indicator.
Provides visual representation and trading recommendations based on indicator values.

**Usage**

1.Select Cryptocurrency:
Choose a cryptocurrency from the sidebar (BTC, ETH, LTC, DOGE).
2.Select Technical Analysis Type:
Choose an analysis type from the sidebar (Moving Averages, RSI, MACD, Bollinger Bands, EMA, Stochastic Oscillator, Ichimoku Cloud).
3.View Results:
The app displays the candlestick chart for the selected cryptocurrency.
It also provides a detailed technical analysis chart and a textual recommendation based on the selected indicator.


**Installation**

To run the app locally, follow these steps:

1.Clone the repository:

git clone https://github.com/yourusername/crypto-analysis-dashboard.git
cd crypto-analysis-dashboard

2.Install dependencies:

pip install -r requirements.txt

3.Run the app:

streamlit run app.py

**Dependencies:**

Streamlit
yfinance
pandas
plotly

**Contributing**

Feel free to contribute by opening issues or submitting pull requests. For major changes, please open an issue first to discuss what you would like to change.

**License**

This project is licensed under the MIT License.






