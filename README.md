# TradeX---AI-Driven-Crypto-Trading-System

The TradeX - AI-Driven Crypto Trading System is a cutting-edge cryptocurrency trading platform that leverages advanced AI models, machine learning algorithms, and technical analysis to forecast market movements, generate trading strategies, and execute trades automatically. This system integrates LSTM neural networks, ensemble learning models, AI-generated strategies, and sentiment analysis to provide accurate predictions and improve decision-making for cryptocurrency trading.

Features
Market Data Fetching: Uses Yahoo Finance (yfinance) to fetch historical market data for BTC, ETH, ADA, and other altcoins, with technical indicators computed using the ta library.
LSTM Time-Series Model: Implements an LSTM model for price prediction using historical price data and technical indicators.
Ensemble Learning: Combines SVR, Random Forest, and Lasso models with LSTM predictions to enhance model accuracy.
Ichimoku Cloud Analysis: Integrates Ichimoku Cloud and other technical indicators for a visual representation of trading signals.
Backtesting: Provides a simple backtesting implementation for strategy evaluation, allowing comparison between strategy returns and market returns.
AI Strategy Generation: Utilizes GPT-Neo for natural language-based strategy generation and DistilBERT for sentiment analysis, creating an AI-to-AI conversation loop to refine trading strategies.
Prophet Forecasting: Uses Facebook's Prophet to provide future price forecasts based on historical data.
News Sentiment Analysis: Integrates FinBERT to analyze the sentiment of news headlines for enhanced decision-making.

Key Files
ap.py: The main script that contains the core logic, including data fetching, model training, AI-generated strategies, backtesting, and trade execution.
backtest_results.png: Visualizes the cumulative returns from the backtesting process.
feature_correlation.png: Displays the correlation between technical indicators and the closing price.
ichimoku_[symbol].png: Candlestick charts with Ichimoku Cloud analysis for BTC, ETH, and ADA.
model_comparison_[symbol].png: Compares the performance of the LSTM model and ensemble learning models.
predictions_[symbol].png: Displays the actual vs predicted prices for each cryptocurrency.

How It Works
Data Fetching and Preprocessing: Market data for selected cryptocurrencies is fetched using yfinance. Technical indicators are computed, and the dataset is preprocessed for LSTM training.
Model Training: The LSTM model and ensemble models are trained on historical data. Model performance is evaluated using mean squared error (MSE) and mean absolute error (MAE).

AI Strategy Generation: A market summary is created using the latest data and predictions. GPT-Neo generates a trading strategy based on this summary, which is analyzed by DistilBERT for sentiment. The strategy is refined in an AI-to-AI conversation loop.

Trading: Based on the strategy's sentiment, the system automatically places buy/sell orders using the Alpaca API.
Backtesting: A simple moving average crossover strategy is used for backtesting, and the strategy’s returns are plotted against the market returns.

Results
Backtesting: The backtesting results show how the strategy performs over time, compared to the market.
Model Performance: The LSTM and ensemble models are evaluated and plotted to show their prediction accuracy.
AI Strategy: The AI-generated strategy undergoes several refinement rounds before determining whether to place a trade.
