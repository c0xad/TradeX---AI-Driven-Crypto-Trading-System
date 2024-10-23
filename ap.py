import os
import sys
import time
import json
import torch
import logging
import requests
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import VotingRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import (
    RSIIndicator, StochasticOscillator, AwesomeOscillatorIndicator
)
from ta.trend import (
    MACD, SMAIndicator, EMAIndicator, ADXIndicator, CCIIndicator
)
from prophet import Prophet
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    AutoModelForSequenceClassification,
    DistilBertTokenizerFast, DistilBertForSequenceClassification
)
import yfinance as yf  # For historical data
import alpaca_trade_api as tradeapi  # Alpaca API
import mplfinance as mpf  # For advanced plotting

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Set up logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('crypto_trading.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Ensure reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Device setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Configuration
CONFIG = {
    'data': {
        'symbols': ['BTC-USD', 'ETH-USD', 'ADA-USD'],  # Include altcoins
        'start_date': '2017-01-01',
        'end_date': datetime.datetime.now().strftime('%Y-%m-%d'),
        'sequence_length': 60,
        'test_size': 0.2,
        'batch_size': 64
    },
    'lstm': {
        'hidden_size': 1024,
        'num_layers': 3,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'epochs': 50,
        'patience': 10
    },
    'prophet': {
        'changepoint_prior_scale': 0.05,
        'seasonality_prior_scale': 10,
        'forecast_periods': 7
    },
    'llm': {
        'max_length': 1024,
        'temperature': 0.7,
        'top_k': 50,
        'top_p': 0.95,
        'num_rounds': 3
    },
    'alpaca': {
        'api_key': 'YOUR ALPACA API KEY',  # Replace with your Alpaca API Key
        'secret_key': 'YOUR ALPACA SECRET KEY',  # Replace with your Alpaca Secret Key
        'base_url': 'https://paper-api.alpaca.markets'
    }
}

# Initialize Alpaca API
alpaca_api = tradeapi.REST(
    key_id=CONFIG['alpaca']['api_key'],
    secret_key=CONFIG['alpaca']['secret_key'],
    base_url=CONFIG['alpaca']['base_url']
)

# Functions and Classes
def get_market_data(symbol="BTC-USD", start_date='2017-01-01', end_date=None):
    """Fetch historical market data using yfinance."""
    try:
        df = yf.download(symbol, start=start_date, end=end_date)
        df.reset_index(inplace=True)
        df.rename(columns={
            'Date': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adj_close',
            'Volume': 'volume'
        }, inplace=True)
        logger.info(f"Fetched {len(df)} rows of data for {symbol}.")
        return df
    except Exception as e:
        logger.error(f"Error fetching market data for {symbol}: {e}")
        sys.exit(1)

def add_technical_indicators(df):
    """Add technical indicators to the DataFrame."""
    close_prices = df['close']
    high_prices = df['high']
    low_prices = df['low']
    df['rsi'] = RSIIndicator(close=close_prices).rsi()
    macd = MACD(close=close_prices)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    bollinger = BollingerBands(close=close_prices)
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_low'] = bollinger.bollinger_lband()
    df['bb_mean'] = bollinger.bollinger_mavg()
    df['ema_20'] = EMAIndicator(close=close_prices, window=20).ema_indicator()
    df['sma_50'] = SMAIndicator(close=close_prices, window=50).sma_indicator()
    df['sma_200'] = SMAIndicator(close=close_prices, window=200).sma_indicator()
    df['stochastic_oscillator'] = StochasticOscillator(
        high=high_prices, low=low_prices, close=close_prices
    ).stoch()
    df['atr'] = AverageTrueRange(
        high=high_prices, low=low_prices, close=close_prices
    ).average_true_range()
    df['adx'] = ADXIndicator(
        high=high_prices, low=low_prices, close=close_prices
    ).adx()
    df['cci'] = CCIIndicator(
        high=high_prices, low=low_prices, close=close_prices
    ).cci()
    df['ao'] = AwesomeOscillatorIndicator(
        high=high_prices, low=low_prices
    ).awesome_oscillator()
    # Ichimoku Cloud
    df = calculate_advanced_indicators(df)
    logger.debug("Technical indicators added to DataFrame.")
    return df

def calculate_advanced_indicators(df):
    """Calculate advanced technical indicators like Ichimoku Cloud."""
    high_prices = df['high']
    low_prices = df['low']
    close_prices = df['close']
    # Tenkan-sen (Conversion Line)
    nine_period_high = high_prices.rolling(window=9).max()
    nine_period_low = low_prices.rolling(window=9).min()
    df['tenkan_sen'] = (nine_period_high + nine_period_low) / 2
    # Kijun-sen (Base Line)
    twenty_six_period_high = high_prices.rolling(window=26).max()
    twenty_six_period_low = low_prices.rolling(window=26).min()
    df['kijun_sen'] = (twenty_six_period_high + twenty_six_period_low) / 2
    # Senkou Span A (Leading Span A)
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
    # Senkou Span B (Leading Span B)
    fifty_two_period_high = high_prices.rolling(window=52).max()
    fifty_two_period_low = low_prices.rolling(window=52).min()
    df['senkou_span_b'] = ((fifty_two_period_high + fifty_two_period_low) / 2).shift(26)
    # Chikou Span (Lagging Span)
    df['chikou_span'] = close_prices.shift(-26)
    logger.debug("Advanced technical indicators calculated.")
    return df

def handle_missing_data(df):
    """Handle missing data."""
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    logger.debug("Missing data handled in DataFrame.")
    return df

def preprocess_lstm_data(df, sequence_length=60):
    """Preprocess data for LSTM model."""
    features = [
        'close', 'rsi', 'macd', 'macd_signal', 'bb_high',
        'bb_low', 'bb_mean', 'ema_20', 'sma_50', 'sma_200',
        'stochastic_oscillator', 'atr', 'adx', 'cci', 'ao',
        'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span'
    ]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, 0])  # Predict 'close' price
    X = np.array(X)
    y = np.array(y)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=CONFIG['data']['test_size'], shuffle=False
    )
    logger.debug("Data preprocessed for LSTM model.")
    return X_train, X_val, y_train, y_val, scaler

class CryptoDataset(Dataset):
    """Custom Dataset for cryptocurrency data."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.float32).to(device)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    """LSTM model for time series prediction."""
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out.squeeze()

def train_lstm_model(model, train_loader, val_loader):
    """Train the LSTM model."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lstm']['learning_rate'])
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []
    for epoch in range(CONFIG['lstm']['epochs']):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        logger.info(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_lstm_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['lstm']['patience']:
                logger.info("Early stopping")
                break
    model.load_state_dict(torch.load('best_lstm_model.pth'))
    logger.debug("LSTM model training completed.")
    return model, train_losses, val_losses

def evaluate_model(model, data_loader):
    """Evaluate the model."""
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            outputs = model(X_batch)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(y_batch.cpu().numpy())
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    logger.debug("Model evaluation completed.")
    return mse, mae, predictions, actuals

def plot_predictions(predictions, actuals, symbol):
    """Plot predictions vs actuals."""
    plt.figure(figsize=(12, 6))
    plt.plot(actuals, label='Actual Prices')
    plt.plot(predictions, label='Predicted Prices')
    plt.title(f'Actual vs Predicted Prices for {symbol}')
    plt.xlabel('Time')
    plt.ylabel('Normalized Price')
    plt.legend()
    plt.savefig(f'predictions_{symbol.replace("-", "_")}.png')
    plt.close()
    logger.debug(f"Prediction plot saved for {symbol}.")

def plot_candlestick_with_ichimoku(df, symbol):
    """Plot candlestick chart with Ichimoku Cloud."""
    df_ichimoku = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b']]
    df_ichimoku.set_index('timestamp', inplace=True)

    # Create additional plots for Tenkan-sen and Kijun-sen
    apds = [
        mpf.make_addplot(df_ichimoku['tenkan_sen'], color='blue', width=1),
        mpf.make_addplot(df_ichimoku['kijun_sen'], color='red', width=1),
        mpf.make_addplot(df_ichimoku['senkou_span_a'], color='green', width=1),
        mpf.make_addplot(df_ichimoku['senkou_span_b'], color='red', width=1)
    ]

    # Plot configuration
    kwargs = dict(
        type='candle',
        style='charles',
        title=f'{symbol} Price Chart with Ichimoku Cloud',
        addplot=apds,
        figsize=(12, 8),
        volume=True,
        panel_ratios=(4, 1),
        savefig=f'ichimoku_{symbol.replace("-", "_")}.png'
    )

    mpf.plot(df_ichimoku, **kwargs)
    logger.debug(f"Ichimoku chart plotted for {symbol}.")

def plot_model_comparisons(actuals, lstm_preds, ensemble_preds, symbol):
    """Plot model predictions for comparison."""
    plt.figure(figsize=(12, 6))
    plt.plot(actuals, label='Actual Prices')
    plt.plot(lstm_preds, label='LSTM Predictions')
    plt.plot(ensemble_preds, label='Ensemble Predictions')
    plt.title(f'Model Comparison for {symbol}')
    plt.xlabel('Time')
    plt.ylabel('Normalized Price')
    plt.legend()
    plt.savefig(f'model_comparison_{symbol.replace("-", "_")}.png')
    plt.close()
    logger.debug(f"Model comparison plot saved for {symbol}.")

def plot_feature_correlation(df, features):
    """Plot correlation of features with the target variable."""
    correlations = []
    target = df['close']
    for feature in features:
        correlation = df[feature].corr(target)
        correlations.append(correlation)
    plt.figure(figsize=(10, 6))
    plt.barh(features, correlations)
    plt.title('Feature Correlation with Close Price')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Features')
    plt.savefig('feature_correlation.png')
    plt.close()
    logger.debug("Feature correlation plot saved.")

def backtest_strategy(df, strategy_function):
    """Simple backtesting implementation."""
    df['signal'] = df.apply(strategy_function, axis=1)
    df['returns'] = df['close'].pct_change()
    df['strategy_returns'] = df['signal'].shift(1) * df['returns']
    cumulative_returns = (1 + df['strategy_returns']).cumprod() - 1
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], cumulative_returns, label='Strategy Returns')
    plt.plot(df['timestamp'], (1 + df['returns']).cumprod() - 1, label='Market Returns')
    plt.title('Backtest Results')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.savefig('backtest_results.png')
    plt.close()
    logger.debug("Backtest results plotted.")
    return cumulative_returns.iloc[-1]

class ProphetModel:
    """Wrapper for Facebook Prophet model."""
    def __init__(self, changepoint_prior_scale, seasonality_prior_scale):
        self.model = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True
        )
    def fit(self, df):
        prophet_df = df[['timestamp', 'close']].rename(columns={'timestamp': 'ds', 'close': 'y'})
        self.model.fit(prophet_df)
        logger.debug("Prophet model fitted.")
    def predict(self, periods):
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        logger.debug("Prophet model prediction completed.")
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def generate_strategy(market_summary, gpt_model, gpt_tokenizer):
    """Generate trading strategy using an open-source LLM."""
    prompt = f"Based on the following market summary, suggest a detailed trading strategy with clear entry and exit points, risk management techniques, and indicators to monitor:\n\n{market_summary}"
    inputs = gpt_tokenizer.encode(prompt, return_tensors='pt').to(device)
    outputs = gpt_model.generate(
        inputs,
        max_length=CONFIG['llm']['max_length'] + inputs.shape[1],
        temperature=CONFIG['llm']['temperature'],
        top_k=CONFIG['llm']['top_k'],
        top_p=CONFIG['llm']['top_p'],
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        pad_token_id=gpt_tokenizer.eos_token_id
    )
    strategy = gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.debug("Strategy generated by the LLM.")
    return strategy

def analyze_strategy(strategy, distilbert_model, distilbert_tokenizer):
    """Analyze the sentiment of the strategy using DistilBERT."""
    inputs = distilbert_tokenizer(
        strategy,
        return_tensors='pt',
        truncation=True,
        max_length=512
    ).to(device)
    outputs = distilbert_model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities).item()
    confidence = probabilities[0][predicted_class].item()
    sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    sentiment = sentiment_map[predicted_class]
    logger.debug("Strategy analyzed by DistilBERT.")
    return sentiment, confidence

def llm_conversation(market_summary, gpt_model, gpt_tokenizer, distilbert_model, distilbert_tokenizer):
    """Conduct AI-to-AI conversation to refine strategy."""
    conversation_history = []
    for round_num in range(CONFIG['llm']['num_rounds']):
        # LLM generates strategy
        strategy = generate_strategy(market_summary, gpt_model, gpt_tokenizer)
        conversation_history.append({
            'speaker': 'LLM',
            'content': strategy
        })
        logger.info(f"LLM Strategy (Round {round_num+1}):\n{strategy}")
        # DistilBERT analyzes strategy
        sentiment, confidence = analyze_strategy(strategy, distilbert_model, distilbert_tokenizer)
        conversation_history.append({
            'speaker': 'DistilBERT',
            'content': f"Sentiment: {sentiment}, Confidence: {confidence:.2f}"
        })
        logger.info(f"DistilBERT Analysis (Round {round_num+1}): Sentiment={sentiment}, Confidence={confidence:.2f}")
        # Check if strategy is acceptable
        if sentiment == 'Positive' and confidence > 0.8:
            logger.info("High confidence positive strategy generated.")
            break
        else:
            # Feedback loop to LLM
            feedback = f"Note: Previous strategy was {sentiment} with confidence {confidence:.2f}. Please improve it by providing more actionable insights."
            market_summary += f"\n{feedback}"
            conversation_history.append({
                'speaker': 'Feedback',
                'content': feedback
            })
            logger.info(f"Feedback provided to LLM: {feedback}")
    logger.debug("AI-to-AI conversation completed.")
    return strategy, conversation_history

def fine_tune_distilbert(distilbert_model, distilbert_tokenizer):
    """Fine-tune DistilBERT on a custom dataset."""
    # Sample training data
    training_data = [
        ("This strategy is too risky and likely to result in losses.", 0),
        ("The strategy seems acceptable but could be improved.", 1),
        ("This is a solid strategy with high potential for gains.", 2),
        ("Avoid entering the market now due to high volatility.", 0),
        ("Consider holding your position and monitor key indicators.", 1),
        ("Strong buy signal detected, proceed with the investment.", 2),
        # Add more examples as needed
    ]
    # Prepare data
    inputs = distilbert_tokenizer(
        [text for text, label in training_data],
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=128
    )
    labels = torch.tensor([label for text, label in training_data]).to(device)
    dataset = torch.utils.data.TensorDataset(
        inputs['input_ids'].to(device),
        inputs['attention_mask'].to(device),
        labels
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    # Training loop
    optimizer = optim.AdamW(distilbert_model.parameters(), lr=5e-5)
    distilbert_model.train()
    for epoch in range(5):  # Increased epochs for better fine-tuning
        total_loss = 0
        for batch in loader:
            input_ids, attention_mask, labels = batch
            optimizer.zero_grad()
            outputs = distilbert_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        logger.info(f"Fine-tuning DistilBERT - Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    distilbert_model.eval()
    logger.debug("DistilBERT fine-tuning completed.")

def analyze_news_sentiment(news_headlines, sentiment_model, tokenizer):
    """Analyze sentiment of news headlines using FinBERT."""
    sentiments = []
    for headline in news_headlines:
        inputs = tokenizer(
            headline,
            return_tensors='pt',
            truncation=True,
            max_length=128,
            padding=True
        ).to(device)
        outputs = sentiment_model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities).item()
        sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        sentiment = sentiment_map[predicted_class]
        sentiments.append(sentiment)
    sentiment_counts = pd.Series(sentiments).value_counts(normalize=True)
    logger.debug("News sentiment analysis completed.")
    return sentiment_counts.to_dict()

def fetch_news_headlines(symbol):
    """Fetch recent news headlines related to the symbol."""
    # Placeholder function - Normally you'd use an API to fetch news
    # Since we cannot use additional libraries, we'll simulate this
    sample_headlines = [
        f"{symbol} shows strong bullish momentum.",
        f"Analysts predict {symbol} price surge.",
        f"{symbol} faces regulatory challenges.",
        f"Investors are cautious about {symbol}."
    ]
    return sample_headlines

class ModelEnsemble:
    """Ensemble model combining multiple algorithms."""
    def __init__(self, lstm_model):
        self.models = {
            'svr': SVR(kernel='rbf'),
            'rf': RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED),
            'lasso': LassoCV(cv=5, random_state=RANDOM_SEED)
        }
        self.lstm_model = lstm_model  # Pre-trained LSTM model
        self.ensemble = None

    def fit(self, X, y):
        # Train individual models
        estimators = []
        X_flat = X.reshape(X.shape[0], -1)  # Flatten for sklearn models
        for name, model in self.models.items():
            model.fit(X_flat, y)
            estimators.append((name, model))
        # LSTM predictions as a feature
        lstm_preds = self.lstm_model(torch.tensor(X, dtype=torch.float32).to(device)).detach().cpu().numpy()
        # Combine features
        X_ensemble = np.hstack([model.predict(X_flat).reshape(-1, 1) for name, model in self.models.items()])
        X_ensemble = np.hstack([X_ensemble, lstm_preds.reshape(-1, 1)])
        # Create and train voting ensemble
        self.ensemble = VotingRegressor(estimators=[
            (name, model) for name, model in self.models.items()
        ])
        self.ensemble.fit(X_flat, y)
        logger.debug("Ensemble model trained.")

    def predict(self, X):
        X_flat = X.reshape(X.shape[0], -1)
        predictions = self.ensemble.predict(X_flat)
        return predictions

def place_trade(symbol, quantity, side):
    """Place a trade using Alpaca API."""
    try:
        order = alpaca_api.submit_order(
            symbol=symbol,
            qty=quantity,
            side=side,
            type='market',
            time_in_force='gtc'
        )
        logger.info(f"Placed {side} order for {quantity} units of {symbol}.")
        return order
    except Exception as e:
        logger.error(f"Error placing trade: {e}")
        return None

def main():
    # Load models and tokenizers
    gpt_tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125M')
    gpt_model = AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neo-125M').to(device)
    distilbert_tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    distilbert_model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=3
    ).to(device)
    # Load FinBERT for sentiment analysis
    finbert_model = AutoModelForSequenceClassification.from_pretrained(
        'yiyanghkust/finbert-tone',
        num_labels=3
    ).to(device)
    finbert_tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')
    # Fine-tune DistilBERT
    fine_tune_distilbert(distilbert_model, distilbert_tokenizer)
    # Process each symbol (including altcoins)
    for symbol in CONFIG['data']['symbols']:
        logger.info(f"Processing symbol: {symbol}")
        # Fetch market data
        df = get_market_data(
            symbol=symbol,
            start_date=CONFIG['data']['start_date'],
            end_date=CONFIG['data']['end_date']
        )
        # Add technical indicators
        df = add_technical_indicators(df)
        df = handle_missing_data(df)
        # Preprocess data
        X_train, X_val, y_train, y_val, scaler = preprocess_lstm_data(
            df, sequence_length=CONFIG['data']['sequence_length']
        )
        # Create datasets and loaders
        train_dataset = CryptoDataset(X_train, y_train)
        val_dataset = CryptoDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=CONFIG['data']['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=CONFIG['data']['batch_size'], shuffle=False)
        # Initialize and train LSTM model
        lstm_model = LSTMModel(
            input_size=X_train.shape[2],
            hidden_size=CONFIG['lstm']['hidden_size'],
            num_layers=CONFIG['lstm']['num_layers'],
            dropout=CONFIG['lstm']['dropout']
        ).to(device)
        lstm_model, train_losses, val_losses = train_lstm_model(lstm_model, train_loader, val_loader)
        # Evaluate LSTM model
        mse_lstm, mae_lstm, lstm_predictions, actuals = evaluate_model(lstm_model, val_loader)
        logger.info(f"LSTM Evaluation for {symbol} - MSE: {mse_lstm:.6f}, MAE: {mae_lstm:.6f}")
        # Initialize and train ensemble model
        ensemble_model = ModelEnsemble(lstm_model)
        ensemble_model.fit(X_train, y_train)
        ensemble_predictions = ensemble_model.predict(X_val)
        mse_ensemble = mean_squared_error(y_val, ensemble_predictions)
        mae_ensemble = mean_absolute_error(y_val, ensemble_predictions)
        logger.info(f"Ensemble Evaluation for {symbol} - MSE: {mse_ensemble:.6f}, MAE: {mae_ensemble:.6f}")
        # Plot predictions
        plot_predictions(lstm_predictions, actuals, symbol)
        plot_model_comparisons(actuals, lstm_predictions, ensemble_predictions, symbol)
        # Plot candlestick chart with Ichimoku Cloud
        plot_candlestick_with_ichimoku(df, symbol)
        # Plot feature correlation
        features = [
            'rsi', 'macd', 'macd_signal', 'bb_high',
            'bb_low', 'bb_mean', 'ema_20', 'sma_50', 'sma_200',
            'stochastic_oscillator', 'atr', 'adx', 'cci', 'ao',
            'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span'
        ]
        plot_feature_correlation(df, features)
        # Use Prophet for forecasting
        prophet_model = ProphetModel(
            changepoint_prior_scale=CONFIG['prophet']['changepoint_prior_scale'],
            seasonality_prior_scale=CONFIG['prophet']['seasonality_prior_scale']
        )
        prophet_model.fit(df)
        forecast = prophet_model.predict(CONFIG['prophet']['forecast_periods'])
        # Generate market summary
        latest_data = df.iloc[-1]
        lstm_prediction = scaler.inverse_transform(
            np.concatenate((
                [[lstm_predictions[-1]]],
                np.zeros((1, X_val.shape[2]-1))
            ), axis=1)
        )[0][0]
        ensemble_prediction = scaler.inverse_transform(
            np.concatenate((
                [[ensemble_predictions[-1]]],
                np.zeros((1, X_val.shape[2]-1))
            ), axis=1)
        )[0][0]
        market_summary = f"""
        Symbol: {symbol}
        Date: {latest_data['timestamp'].date()}
        Close Price: ${latest_data['close']:.2f}
        RSI: {latest_data['rsi']:.2f}
        MACD: {latest_data['macd']:.2f}
        Bollinger Bands: High ${latest_data['bb_high']:.2f}, Low ${latest_data['bb_low']:.2f}
        LSTM Prediction: ${lstm_prediction:.2f}
        Ensemble Prediction: ${ensemble_prediction:.2f}
        Prophet Forecast: ${forecast['yhat'].iloc[-1]:.2f}
        """
        # Fetch and analyze news sentiment
        news_headlines = fetch_news_headlines(symbol)
        news_sentiment = analyze_news_sentiment(news_headlines, finbert_model, finbert_tokenizer)
        market_summary += f"\nNews Sentiment: {news_sentiment}"
        # AI-to-AI conversation
        strategy, conversation_history = llm_conversation(
            market_summary, gpt_model, gpt_tokenizer,
            distilbert_model, distilbert_tokenizer
        )
        # Log the conversation history
        logger.info(f"AI-to-AI Conversation for {symbol}:")
        for entry in conversation_history:
            logger.info(f"{entry['speaker']}: {entry['content']}")
        # Decide on trading action based on strategy sentiment
        sentiment, confidence = analyze_strategy(strategy, distilbert_model, distilbert_tokenizer)
        if sentiment == 'Positive' and confidence > 0.8:
            # Place a buy order
            place_trade(symbol.replace('-USD', ''), quantity=1, side='buy')
        elif sentiment == 'Negative' and confidence > 0.8:
            # Place a sell order
            place_trade(symbol.replace('-USD', ''), quantity=1, side='sell')
        else:
            logger.info(f"No confident action for {symbol}. Strategy sentiment: {sentiment} ({confidence:.2f})")
        # Backtesting (Simple example using moving average crossover)
        def simple_strategy(row):
            if row['sma_50'] > row['sma_200']:
                return 1  # Buy signal
            else:
                return 0  # Sell signal
        cumulative_return = backtest_strategy(df, simple_strategy)
        logger.info(f"Cumulative Return for {symbol}: {cumulative_return:.2%}")
        # Save results
        results = {
            'timestamp': datetime.datetime.now().isoformat(),
            'symbol': symbol,
            'market_summary': market_summary,
            'news_sentiment': news_sentiment,
            'conversation_history': conversation_history,
            'final_strategy': strategy,
            'strategy_sentiment': {'sentiment': sentiment, 'confidence': confidence},
            'lstm_metrics': {'mse': f"{mse_lstm:.8f}", 'mae': f"{mae_lstm:.8f}"},
            'ensemble_metrics': {'mse': f"{mse_ensemble:.8f}", 'mae': f"{mae_ensemble:.8f}"},
            'cumulative_return': cumulative_return,
            'forecast': forecast.tail(5).to_dict(orient='records')
        }
        with open(f'trading_analysis_results_{symbol.replace("-", "_")}.json', 'w') as f:
            json.dump(results, f, indent=4, default=str)
        logger.info(f"Results saved for {symbol}.")
    logger.info("Processing completed for all symbols.")

if __name__ == "__main__":
    main()
