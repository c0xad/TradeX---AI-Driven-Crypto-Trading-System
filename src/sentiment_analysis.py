import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import torch
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional, Union
from datetime import datetime, timedelta
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from scipy.signal import savgol_filter
import praw
from textblob import TextBlob
import asyncpraw
from collections import defaultdict
import aiohttp
import json
from scipy.stats import entropy, skew, kurtosis
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from collections import defaultdict
import warnings
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mutual_info_score
import torch
import torch.nn as nn
from torch.nn import functional as F
from statsmodels.tsa.stattools import adfuller
from sklearn.mixture import GaussianMixture

@dataclass
class SentimentConfig:
    base_decay_rate: float = 0.95
    max_age_days: int = 7
    min_confidence: float = 0.3
    impact_threshold: float = 0.1
    smoothing_window: int = 12
    correlation_threshold: float = 0.3
    entropy_threshold: float = 0.7
    price_correlation_window: int = 48
    volatility_adjustment_factor: float = 0.15
    regime_detection_window: int = 168
    momentum_threshold: float = 0.25
    mean_reversion_threshold: float = 0.3
    vol_regime_threshold: float = 1.5
    gmm_components: int = 3
    minimum_data_points: int = 100
    dynamic_decay_adjustment: bool = True

class PriceActionMetrics:
    def __init__(self, config: SentimentConfig):
        self.config = config
        self.regime_classifier = GaussianMixture(
            n_components=config.gmm_components,
            covariance_type='full',
            random_state=42
        )
        self.volatility_states = []
        self.momentum_states = []
        
    def calculate_price_metrics(self, price_data: pd.DataFrame) -> Dict[str, float]:
        returns = np.log(price_data['close']).diff().dropna()
        
        # Calculate volatility regimes using GMM
        vol = returns.rolling(self.config.regime_detection_window).std()
        vol_features = vol.values.reshape(-1, 1)
        if len(vol_features) >= self.config.minimum_data_points:
            self.volatility_states = self.regime_classifier.fit_predict(vol_features)
        
        # Calculate momentum and mean reversion signals
        momentum = self._calculate_momentum_factor(price_data)
        mean_reversion = self._calculate_mean_reversion_factor(price_data)
        
        # Calculate market efficiency ratio
        mer = self._calculate_market_efficiency_ratio(price_data)
        
        # Detect price breakouts
        breakout_strength = self._calculate_breakout_strength(price_data)
        
        return {
            'volatility_regime': np.mean(self.volatility_states[-10:]) if len(self.volatility_states) > 0 else 0,
            'momentum_factor': momentum,
            'mean_reversion_factor': mean_reversion,
            'market_efficiency': mer,
            'breakout_strength': breakout_strength,
            'volatility_adjusted_returns': self._calculate_volatility_adjusted_returns(returns)
        }
    
    def _calculate_momentum_factor(self, price_data: pd.DataFrame) -> float:
        returns = price_data['close'].pct_change()
        short_ma = returns.rolling(12).mean()
        long_ma = returns.rolling(26).mean()
        macd = short_ma - long_ma
        signal = macd.rolling(9).mean()
        momentum = (macd - signal).iloc[-1]
        return np.tanh(momentum)  # Normalize to [-1, 1]
    
    def _calculate_mean_reversion_factor(self, price_data: pd.DataFrame) -> float:
        returns = price_data['close'].pct_change()
        zscore = (returns - returns.rolling(20).mean()) / returns.rolling(20).std()
        reversion_strength = -zscore.iloc[-1]  # Negative zscore indicates mean reversion
        return np.tanh(reversion_strength)
    
    def _calculate_market_efficiency_ratio(self, price_data: pd.DataFrame) -> float:
        price_path = np.abs(price_data['close'].diff()).sum()
        price_change = np.abs(price_data['close'].iloc[-1] - price_data['close'].iloc[0])
        return price_change / price_path if price_path != 0 else 0
    
    def _calculate_breakout_strength(self, price_data: pd.DataFrame) -> float:
        bb_std = 2
        rolling_mean = price_data['close'].rolling(20).mean()
        rolling_std = price_data['close'].rolling(20).std()
        upper_band = rolling_mean + bb_std * rolling_std
        lower_band = rolling_mean - bb_std * rolling_std
        
        latest_price = price_data['close'].iloc[-1]
        if (latest_price > upper_band.iloc[-1]):
            return (latest_price - upper_band.iloc[-1]) / rolling_std.iloc[-1]
        elif (latest_price < lower_band.iloc[-1]):
            return (latest_price - lower_band.iloc[-1]) / rolling_std.iloc[-1]
        return 0
    
    def _calculate_volatility_adjusted_returns(self, returns: pd.Series) -> float:
        vol = returns.rolling(20).std()
        return (returns / vol).mean() if not vol.empty else 0

class EnhancedSentimentAnalyzer:
    def __init__(self, config: Optional[SentimentConfig] = None):
        self.config = config or SentimentConfig()
        
        # Initialize sentiment memory with temporal features
        self.sentiment_memory = defaultdict(lambda: {
            'scores': [],
            'timestamps': [],
            'confidences': [],
            'impacts': [],
            'sources': []
        })
        
        # Enhanced event patterns with impact correlation tracking
        self.event_patterns = self._initialize_event_patterns()
        
        # Initialize visualization components
        self.fig = None
        self.sentiment_graph = nx.DiGraph()
        
        # Setup neural attention mechanism for temporal weighting
        self.temporal_attention = TemporalAttention(
            input_dim=4,  # sentiment, confidence, impact, time_delta
            hidden_dim=32
        )
        
        # Initialize metrics tracking
        self.performance_metrics = defaultdict(list)
        self.correlation_history = defaultdict(list)
        
        # Setup adaptive thresholds
        self.adaptive_thresholds = self._initialize_adaptive_thresholds()
        
        # Initialize price action metrics
        self.price_metrics = PriceActionMetrics(self.config)
        self.sentiment_regimes = defaultdict(list)
        self.correlation_matrices = defaultdict(list)

    def _initialize_event_patterns(self) -> Dict:
        """Initialize enhanced event patterns with impact tracking"""
        return {
            'major_partnership': {
                'keywords': {'partnership', 'collaboration', 'alliance', 'strategic'},
                'impact_score': 1.5,
                'decay_rate': 0.92,
                'confidence_threshold': 0.7,
                'correlation_history': [],
                'performance_metrics': defaultdict(list)
            },
            'technical_milestone': {
                'keywords': {'mainnet', 'upgrade', 'protocol', 'scaling', 'tps'},
                'impact_score': 1.3,
                'decay_rate': 0.94,
                'confidence_threshold': 0.65,
                'correlation_history': [],
                'performance_metrics': defaultdict(list)
            },
            'security_event': {
                'keywords': {'hack', 'exploit', 'vulnerability', 'audit', 'security'},
                'impact_score': -1.8,
                'decay_rate': 0.85,
                'confidence_threshold': 0.8,
                'correlation_history': [],
                'performance_metrics': defaultdict(list)
            },
            'adoption_milestone': {
                'keywords': {'adoption', 'users', 'tvl', 'volume', 'growth'},
                'impact_score': 1.4,
                'decay_rate': 0.93,
                'confidence_threshold': 0.6,
                'correlation_history': [],
                'performance_metrics': defaultdict(list)
            }
        }

    async def update_sentiment(self, symbol: str, new_data: Dict, 
                             price_data: pd.DataFrame) -> Dict[str, float]:
        """Update sentiment with temporal decay and impact tracking"""
        current_time = datetime.now()
        
        # Calculate price metrics
        price_metrics = self.price_metrics.calculate_price_metrics(price_data)
        
        # Update temporal features with price metrics
        temporal_features = self._prepare_temporal_features(symbol, current_time, price_metrics)
        attention_weights = self.temporal_attention(temporal_features)
        
        # Process sentiment with price action influence
        processed_sentiment = self._process_new_sentiment(new_data, attention_weights, price_metrics)
        
        # Update sentiment memory and correlation matrices
        self._update_sentiment_memory(symbol, processed_sentiment, current_time)
        self._update_correlation_matrices(symbol, price_data)
        
        # Calculate regime-aware sentiment
        aggregated_sentiment = self._calculate_regime_aware_sentiment(symbol, price_metrics)
        
        return aggregated_sentiment

    def _prepare_temporal_features(self, symbol: str, 
                                 current_time: datetime, price_metrics: Dict[str, float]) -> torch.Tensor:
        """Prepare temporal features for attention mechanism"""
        memory = self.sentiment_memory[symbol]
        if not memory['timestamps']:
            return torch.zeros((1, 4))
        
        features = []
        for score, timestamp, conf, impact in zip(
            memory['scores'],
            memory['timestamps'],
            memory['confidences'],
            memory['impacts']
        ):
            time_delta = (current_time - timestamp).total_seconds() / 86400  # Convert to days
            features.append([score, conf, impact, time_delta])
            
        return torch.tensor(features, dtype=torch.float32)

    def _process_new_sentiment(self, new_data: Dict, 
                             attention_weights: torch.Tensor, price_metrics: Dict[str, float]) -> Dict:
        """Process new sentiment data with attention weights"""
        # Extract base sentiment
        base_sentiment = new_data['sentiment_score']
        
        # Detect relevant events
        events = self._detect_events(new_data['text'])
        
        # Calculate event-adjusted sentiment
        event_sentiment = self._calculate_event_sentiment(events, base_sentiment)
        
        # Apply attention weighting
        weighted_sentiment = self._apply_attention_weights(
            event_sentiment, attention_weights
        )
        
        # Calculate confidence and impact scores
        confidence = self._calculate_confidence(new_data, events)
        impact = self._calculate_impact_score(weighted_sentiment, events)
        
        return {
            'sentiment': weighted_sentiment,
            'confidence': confidence,
            'impact': impact,
            'events': events,
            'source': new_data.get('source', 'unknown')
        }

    def _detect_events(self, text: str) -> List[Dict]:
        """Detect sentiment events with enhanced pattern matching"""
        detected_events = []
        
        for event_type, pattern in self.event_patterns.items():
            # Calculate keyword matches
            keyword_matches = sum(1 for keyword in pattern['keywords'] 
                                if keyword in text.lower())
            
            if keyword_matches > 0:
                # Calculate match confidence
                confidence = keyword_matches / len(pattern['keywords'])
                
                if confidence >= pattern['confidence_threshold']:
                    # Calculate historical performance
                    performance_score = np.mean(
                        pattern['performance_metrics']['prediction_accuracy'][-10:]
                    ) if pattern['performance_metrics']['prediction_accuracy'] else 0.5
                    
                    detected_events.append({
                        'type': event_type,
                        'confidence': confidence,
                        'impact_score': pattern['impact_score'] * performance_score,
                        'decay_rate': pattern['decay_rate']
                    })
                    
        return detected_events

    def visualize_sentiment_trends(self, symbol: str, price_data: pd.DataFrame,
                                 window_size: int = 24) -> None:
        """Create interactive visualization of sentiment trends"""
        memory = self.sentiment_memory[symbol]
        
        # Create figure with secondary y-axis
        fig = make_subplots(rows=3, cols=1,
                           shared_xaxes=True,
                           vertical_spacing=0.05,
                           subplot_titles=('Price vs Sentiment',
                                         'Sentiment Components',
                                         'Impact Analysis'))
        
        # Price data
        fig.add_trace(
            go.Scatter(x=price_data.index,
                      y=price_data['close'],
                      name='Price',
                      line=dict(color='blue')),
            row=1, col=1
        )
        
        # Aggregate sentiment
        dates = [t.strftime('%Y-%m-%d %H:%M:%S') for t in memory['timestamps']]
        
        # Calculate rolling metrics
        sentiment_series = pd.Series(memory['scores'], index=dates)
        impact_series = pd.Series(memory['impacts'], index=dates)
        confidence_series = pd.Series(memory['confidences'], index=dates)
        
        # Add sentiment line
        fig.add_trace(
            go.Scatter(x=dates,
                      y=sentiment_series.rolling(window_size).mean(),
                      name='Sentiment MA',
                      line=dict(color='green')),
            row=1, col=1
        )
        
        # Add sentiment components
        fig.add_trace(
            go.Scatter(x=dates,
                      y=impact_series.rolling(window_size).mean(),
                      name='Impact',
                      line=dict(color='red')),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=dates,
                      y=confidence_series.rolling(window_size).mean(),
                      name='Confidence',
                      line=dict(color='purple')),
            row=2, col=1
        )
        
        # Add correlation analysis
        correlation_window = 100
        rolling_corr = self._calculate_rolling_correlation(
            sentiment_series, price_data['close'], correlation_window
        )
        
        fig.add_trace(
            go.Scatter(x=dates[correlation_window:],
                      y=rolling_corr,
                      name='Price-Sentiment Correlation',
                      line=dict(color='orange')),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=900,
            title_text=f"Sentiment Analysis for {symbol}",
            showlegend=True,
            xaxis3_title="Date",
            yaxis_title="Price",
            yaxis2_title="Sentiment Components",
            yaxis3_title="Correlation"
        )
        
        self.fig = fig
        self.fig.show()

    def _calculate_rolling_correlation(self, sentiment_series: pd.Series,
                                    price_series: pd.Series,
                                    window: int) -> np.ndarray:
        """Calculate rolling correlation between sentiment and price"""
        correlations = []
        
        for i in range(window, len(sentiment_series)):
            sent_window = sentiment_series.iloc[i-window:i]
            price_window = price_series.iloc[i-window:i]
            
            if len(sent_window) == len(price_window):
                corr, _ = pearsonr(sent_window, price_window)
                correlations.append(corr)
            else:
                correlations.append(np.nan)
                
        return np.array(correlations)

    def generate_sentiment_report(self, symbol: str) -> Dict:
        """Generate comprehensive sentiment analysis report"""
        memory = self.sentiment_memory[symbol]
        
        # Calculate basic statistics
        basic_stats = {
            'mean_sentiment': np.mean(memory['scores']),
            'sentiment_volatility': np.std(memory['scores']),
            'sentiment_skew': skew(memory['scores']),
            'sentiment_kurtosis': kurtosis(memory['scores'])
        }
        
        # Calculate event impact statistics
        event_stats = self._calculate_event_statistics(symbol)
        
        # Calculate predictive metrics
        predictive_metrics = {
            'accuracy': np.mean(self.performance_metrics['prediction_accuracy']),
            'precision': np.mean(self.performance_metrics['precision']),
            'recall': np.mean(self.performance_metrics['recall']),
            'f1_score': np.mean(self.performance_metrics['f1_score'])
        }
        
        # Calculate temporal decay effectiveness
        decay_metrics = self._calculate_decay_effectiveness(symbol)
        
        return {
            'basic_stats': basic_stats,
            'event_stats': event_stats,
            'predictive_metrics': predictive_metrics,
            'decay_metrics': decay_metrics,
            'correlation_history': self.correlation_history[symbol]
        }

    def _update_correlation_matrices(self, symbol: str, price_data: pd.DataFrame) -> None:
        memory = self.sentiment_memory[symbol]
        if len(memory['scores']) < self.config.minimum_data_points:
            return
            
        sentiment_series = pd.Series(memory['scores'], index=memory['timestamps'])
        returns = price_data['close'].pct_change()
        
        # Calculate rolling correlation matrix
        data = pd.DataFrame({
            'sentiment': sentiment_series,
            'returns': returns,
            'volatility': returns.rolling(20).std(),
            'momentum': returns.rolling(12).mean()
        }).dropna()
        
        correlation_matrix = data.corr()
        self.correlation_matrices[symbol].append(correlation_matrix)
        
        # Update sentiment regimes based on correlation structure
        if len(self.correlation_matrices[symbol]) > 5:
            regime = self._detect_sentiment_regime(symbol)
            self.sentiment_regimes[symbol].append(regime)
    
    def _detect_sentiment_regime(self, symbol: str) -> int:
        recent_correlations = np.array([matrix.values for matrix in self.correlation_matrices[symbol][-5:]])
        regime_features = recent_correlations.reshape(-1, 16)  # 4x4 matrix flattened
        
        if len(regime_features) >= self.config.minimum_data_points:
            return self.price_metrics.regime_classifier.fit_predict(regime_features)[-1]
        return 0
    
    def _calculate_regime_aware_sentiment(self, symbol: str, price_metrics: Dict[str, float]) -> Dict[str, float]:
        base_sentiment = super()._calculate_aggregated_sentiment(symbol)
        
        # Adjust sentiment based on market regime
        regime_adjustment = self._calculate_regime_adjustment(symbol, price_metrics)
        
        # Calculate final sentiment scores
        adjusted_sentiment = {
            'sentiment_score': base_sentiment['sentiment_score'] * (1 + regime_adjustment),
            'confidence': base_sentiment['confidence'] * price_metrics['market_efficiency'],
            'impact': base_sentiment['impact'] * (1 + price_metrics['breakout_strength']),
            'regime': self.sentiment_regimes[symbol][-1] if self.sentiment_regimes[symbol] else 0,
            'price_metrics': price_metrics
        }
        
        return adjusted_sentiment
    
    def _calculate_regime_adjustment(self, symbol: str, price_metrics: Dict[str, float]) -> float:
        if not self.sentiment_regimes[symbol]:
            return 0
            
        current_regime = self.sentiment_regimes[symbol][-1]
        volatility_factor = price_metrics['volatility_regime']
        momentum_factor = price_metrics['momentum_factor']
        
        # Calculate regime-specific adjustment
        if current_regime == 0:  # Low impact regime
            adjustment = momentum_factor * 0.5
        elif current_regime == 1:  # Medium impact regime
            adjustment = momentum_factor * 1.0
        else:  # High impact regime
            adjustment = momentum_factor * 1.5
            
        return adjustment * (1 + volatility_factor * self.config.volatility_adjustment_factor)

class TemporalAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=0)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention(x)