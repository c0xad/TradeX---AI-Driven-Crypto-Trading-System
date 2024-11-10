import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.stats import linregress
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime

@dataclass
class TimeframeConfig:
    short: int
    medium: int
    long: int
    weights: Dict[str, float]

@dataclass
class MarketRegime:
    volatility: float
    trend_strength: float
    momentum: float
    regime_type: str

class EnhancedTechnicalIndicators:
    def __init__(self):
        self.timeframes = {
            'short': TimeframeConfig(5, 10, 20, {'weight': 0.4, 'volatility_adj': 1.2}),
            'medium': TimeframeConfig(10, 20, 40, {'weight': 0.35, 'volatility_adj': 1.0}),
            'long': TimeframeConfig(20, 40, 80, {'weight': 0.25, 'volatility_adj': 0.8})
        }
        self.scaler = StandardScaler()
        self.regime_thresholds = {
            'volatility': {'low': 0.5, 'high': 1.5},
            'trend': {'weak': 0.3, 'strong': 0.7}
        }
        
        # Add Ichimoku parameters with adaptive spans
        self.ichimoku_params = {
            'tenkan': 9,
            'kijun': 26,
            'senkou_span_b': 52,
            'displacement': 26,
            'cloud_offset': 26
        }
        
        # Add visualization parameters
        self.plot_config = {
            'cloud_alpha': 0.3,
            'line_width': 1.5,
            'marker_size': 4,
            'grid_alpha': 0.2
        }
        
        # Initialize regime visualization
        self.fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                'Price & Hybrid Indicators',
                'Regime Classification',
                'Momentum Analysis'
            )
        )

    def calculate_multi_timeframe_macd(self, prices: pd.Series) -> Dict[str, Tuple[pd.Series, pd.Series, pd.Series]]:
        mtf_macd = {}
        
        for tf_name, tf_config in self.timeframes.items():
            short_ema = prices.ewm(span=tf_config.short, adjust=False).mean()
            long_ema = prices.ewm(span=tf_config.long, adjust=False).mean()
            macd = short_ema - long_ema
            signal = macd.ewm(span=9, adjust=False).mean()
            
            # Enhanced smoothing with Savitzky-Golay filter
            macd_smooth = pd.Series(
                savgol_filter(macd, window_length=11, polyorder=3),
                index=macd.index
            )
            signal_smooth = pd.Series(
                savgol_filter(signal, window_length=11, polyorder=3),
                index=signal.index
            )
            histogram = macd_smooth - signal_smooth
            
            mtf_macd[tf_name] = (macd_smooth, signal_smooth, histogram)
            
        return mtf_macd

    def calculate_directional_movement_index(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                                          period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()

        # Calculate Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        pos_dm = ((up_move > down_move) & (up_move > 0)) * up_move
        neg_dm = ((down_move > up_move) & (down_move > 0)) * down_move
        
        # Smooth DM values
        pos_di = 100 * pos_dm.ewm(span=period, adjust=False).mean() / atr
        neg_di = 100 * neg_dm.ewm(span=period, adjust=False).mean() / atr
        
        # Calculate ADX
        dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
        adx = dx.ewm(span=period, adjust=False).mean()
        
        return pos_di, neg_di, adx

    def calculate_hybrid_bands(self, prices: pd.Series, macd: pd.Series, 
                             window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        # Combine MACD and price for enhanced volatility detection
        combined_series = (self.scaler.fit_transform(prices.values.reshape(-1, 1)) + 
                         self.scaler.fit_transform(macd.values.reshape(-1, 1))) / 2
        
        # Calculate dynamic standard deviation based on MACD trend
        rolling_std = pd.Series(combined_series.flatten(), index=prices.index).rolling(window=window).std()
        macd_trend = macd.rolling(window=window).mean()
        dynamic_std = rolling_std * (1 + abs(macd_trend))
        
        # Calculate bands with dynamic width
        middle_band = prices.rolling(window=window).mean()
        upper_band = middle_band + (dynamic_std * num_std)
        lower_band = middle_band - (dynamic_std * num_std)
        
        return upper_band, middle_band, lower_band

    def calculate_adaptive_rsi(self, prices: pd.Series, initial_window: int = 14) -> pd.Series:
        # Calculate price volatility
        volatility = prices.pct_change().rolling(window=initial_window).std()
        
        # Adjust RSI window based on volatility
        dynamic_window = (initial_window * (1 + volatility)).astype(int)
        dynamic_window = dynamic_window.clip(lower=10, upper=30)
        
        # Calculate RSI with dynamic window
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Use dynamic windows for average gain/loss
        avg_gain = gain.rolling(window=dynamic_window, min_periods=1).mean()
        avg_loss = loss.rolling(window=dynamic_window, min_periods=1).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def detect_market_regime(self, prices: pd.Series, window: int = 20) -> MarketRegime:
        returns = prices.pct_change()
        
        # Calculate volatility using Garman-Klass estimator
        volatility = np.sqrt(
            (returns ** 2).rolling(window=window).mean() * 252
        )
        
        # Calculate trend strength using linear regression
        x = np.arange(len(prices[-window:]))
        slope, _, r_value, _, _ = linregress(x, prices[-window:])
        trend_strength = abs(r_value)
        
        # Calculate momentum using ROC
        momentum = (prices / prices.shift(window) - 1).iloc[-1]
        
        # Determine regime type
        regime_type = self._classify_regime(volatility.iloc[-1], trend_strength, momentum)
        
        return MarketRegime(
            volatility=volatility.iloc[-1],
            trend_strength=trend_strength,
            momentum=momentum,
            regime_type=regime_type
        )

    def _classify_regime(self, volatility: float, trend_strength: float, momentum: float) -> str:
        if volatility > self.regime_thresholds['volatility']['high']:
            if trend_strength > self.regime_thresholds['trend']['strong']:
                return 'trending_volatile'
            return 'choppy_volatile'
        elif volatility < self.regime_thresholds['volatility']['low']:
            if trend_strength < self.regime_thresholds['trend']['weak']:
                return 'ranging_low_vol'
            return 'trending_low_vol'
        else:
            if trend_strength > self.regime_thresholds['trend']['strong']:
                return 'trending_normal'
            return 'mixed'

    def generate_adaptive_signals(self, prices: pd.Series, high: pd.Series, low: pd.Series, 
                                close: pd.Series) -> Dict[str, float]:
        # Get market regime
        regime = self.detect_market_regime(prices)
        
        # Calculate multi-timeframe indicators
        mtf_macd = self.calculate_multi_timeframe_macd(prices)
        pos_di, neg_di, adx = self.calculate_directional_movement_index(high, low, close)
        rsi = self.calculate_adaptive_rsi(prices)
        
        # Calculate hybrid bands using primary timeframe MACD
        upper_band, middle_band, lower_band = self.calculate_hybrid_bands(
            prices, mtf_macd['medium'][0]
        )
        
        # Generate weighted signals based on market regime
        signals = {}
        
        # Trend signals
        trend_score = self._calculate_trend_score(mtf_macd, pos_di, neg_di, adx, regime)
        
        # Momentum signals
        momentum_score = self._calculate_momentum_score(rsi, mtf_macd, regime)
        
        # Mean reversion signals
        reversion_score = self._calculate_reversion_score(
            prices, upper_band, lower_band, middle_band, regime
        )
        
        # Combine signals with regime-based weights
        signals['trend'] = trend_score
        signals['momentum'] = momentum_score
        signals['mean_reversion'] = reversion_score
        signals['composite'] = self._calculate_composite_score(signals, regime)
        
        return signals

    def _calculate_trend_score(self, mtf_macd: Dict, pos_di: pd.Series, 
                             neg_di: pd.Series, adx: pd.Series, 
                             regime: MarketRegime) -> float:
        # Weight signals based on market regime
        regime_weights = {
            'trending_volatile': {'macd': 0.4, 'dmi': 0.6},
            'choppy_volatile': {'macd': 0.3, 'dmi': 0.7},
            'trending_normal': {'macd': 0.5, 'dmi': 0.5},
            'ranging_low_vol': {'macd': 0.6, 'dmi': 0.4},
            'trending_low_vol': {'macd': 0.7, 'dmi': 0.3},
            'mixed': {'macd': 0.5, 'dmi': 0.5}
        }
        
        weights = regime_weights[regime.regime_type]
        
        # Calculate MACD score across timeframes
        macd_score = 0
        for tf, (macd, signal, hist) in mtf_macd.items():
            tf_weight = self.timeframes[tf].weights['weight']
            macd_score += tf_weight * (
                1 if macd.iloc[-1] > signal.iloc[-1] else -1
            ) * abs(hist.iloc[-1])
            
        # Calculate DMI score
        dmi_score = (1 if pos_di.iloc[-1] > neg_di.iloc[-1] else -1) * (adx.iloc[-1] / 100)
        
        # Combine scores
        return weights['macd'] * macd_score + weights['dmi'] * dmi_score

    def _calculate_momentum_score(self, rsi: pd.Series, mtf_macd: Dict, 
                                regime: MarketRegime) -> float:
        # Adjust RSI thresholds based on regime
        rsi_thresholds = {
            'trending_volatile': {'oversold': 30, 'overbought': 70},
            'choppy_volatile': {'oversold': 40, 'overbought': 60},
            'trending_normal': {'oversold': 35, 'overbought': 65},
            'ranging_low_vol': {'oversold': 45, 'overbought': 55},
            'trending_low_vol': {'oversold': 40, 'overbought': 60},
            'mixed': {'oversold': 40, 'overbought': 60}
        }
        
        thresholds = rsi_thresholds[regime.regime_type]
        
        # Calculate RSI score
        rsi_value = rsi.iloc[-1]
        if rsi_value < thresholds['oversold']:
            rsi_score = 1 * (thresholds['oversold'] - rsi_value) / thresholds['oversold']
        elif rsi_value > thresholds['overbought']:
            rsi_score = -1 * (rsi_value - thresholds['overbought']) / (100 - thresholds['overbought'])
        else:
            rsi_score = 0
            
        # Combine with MACD momentum
        macd_momentum = sum(
            self.timeframes[tf].weights['weight'] * hist.iloc[-1]
            for tf, (_, _, hist) in mtf_macd.items()
        )
        
        # Weight combination based on regime
        if regime.regime_type in ['trending_volatile', 'trending_normal']:
            return 0.4 * rsi_score + 0.6 * np.sign(macd_momentum)
        else:
            return 0.6 * rsi_score + 0.4 * np.sign(macd_momentum)

    def _calculate_reversion_score(self, prices: pd.Series, upper: pd.Series, 
                                 lower: pd.Series, middle: pd.Series, 
                                 regime: MarketRegime) -> float:
        # Calculate price position within bands
        price = prices.iloc[-1]
        band_position = (price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])
        
        # Calculate distance from middle band
        middle_distance = (price - middle.iloc[-1]) / middle.iloc[-1]
        
        # Adjust scoring based on regime
        if regime.regime_type in ['ranging_low_vol', 'choppy_volatile']:
            # Stronger mean reversion signals
            if band_position < 0.2:
                return 1 * (1 - band_position * 5)
            elif band_position > 0.8:
                return -1 * ((band_position - 0.8) * 5)
            else:
                return -np.sign(middle_distance) * abs(middle_distance) * 2
        else:
            # Weaker mean reversion signals during trends
            if band_position < 0.1:
                return 0.5 * (1 - band_position * 10)
            elif band_position > 0.9:
                return -0.5 * ((band_position - 0.9) * 10)
            else:
                return 0

    def _calculate_composite_score(self, signals: Dict[str, float], 
                                 regime: MarketRegime,
                                 volume_profile: pd.DataFrame) -> float:
        """Enhanced composite score calculation with volume profile"""
        # Get base weights from parent method
        base_score = super()._calculate_composite_score(signals, regime)
        
        # Calculate volume profile influence
        current_price = signals.get('current_price', 0)
        poc_distance = abs(current_price - volume_profile['poc'].iloc[-1])
        volume_influence = 1 - (poc_distance / (volume_profile['va_high'].iloc[-1] - volume_profile['va_low'].iloc[-1]))
        
        # Adjust score based on volume profile
        volume_adjusted_score = base_score * (1 + volume_influence * 0.3)
        
        # Apply non-linear transformation for extreme regimes
        if abs(volume_adjusted_score) > 0.8:
            volume_adjusted_score = np.sign(volume_adjusted_score) * (
                0.8 + 0.2 * np.tanh(abs(volume_adjusted_score) - 0.8)
            )
        
        return np.clip(volume_adjusted_score, -1, 1)

    def calculate_hybrid_ichimoku_macd(self, prices: pd.Series, high: pd.Series, 
                                     low: pd.Series) -> Dict[str, pd.Series]:
        """Calculate hybrid Ichimoku-MACD indicator"""
        # Calculate Ichimoku components
        tenkan_sen = self._calculate_ichimoku_line(
            high, low, self.ichimoku_params['tenkan']
        )
        kijun_sen = self._calculate_ichimoku_line(
            high, low, self.ichimoku_params['kijun']
        )
        
        # Calculate Cloud components
        senkou_span_a = (tenkan_sen + kijun_sen) / 2
        senkou_span_a = senkou_span_a.shift(self.ichimoku_params['displacement'])
        
        senkou_span_b = self._calculate_ichimoku_line(
            high, low, self.ichimoku_params['senkou_span_b']
        )
        senkou_span_b = senkou_span_b.shift(self.ichimoku_params['displacement'])
        
        # Calculate MACD with adaptive parameters
        macd, signal, hist = self._calculate_adaptive_macd(prices)
        
        # Calculate hybrid signal
        cloud_trend = pd.Series(
            np.where(senkou_span_a > senkou_span_b, 1,
                    np.where(senkou_span_a < senkou_span_b, -1, 0)),
            index=prices.index
        )
        
        # Combine Ichimoku and MACD signals
        hybrid_signal = (
            0.6 * np.sign(prices - kijun_sen) +
            0.4 * np.sign(macd - signal)
        ) * abs(hist) * cloud_trend
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'macd': macd,
            'signal': signal,
            'histogram': hist,
            'hybrid_signal': hybrid_signal,
            'cloud_trend': cloud_trend
        }

    def _calculate_ichimoku_line(self, high: pd.Series, low: pd.Series, 
                                period: int) -> pd.Series:
        """Calculate Ichimoku line with enhanced period adaptation"""
        high_values = high.rolling(window=period).max()
        low_values = low.rolling(window=period).min()
        
        # Apply Savitzky-Golay smoothing
        ichimoku_line = (high_values + low_values) / 2
        return pd.Series(
            savgol_filter(ichimoku_line, window_length=11, polyorder=3),
            index=ichimoku_line.index
        )

    def _calculate_adaptive_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD with adaptive parameters based on volatility"""
        volatility = prices.pct_change().rolling(window=21).std()
        
        # Adjust MACD parameters based on volatility
        fast_period = np.maximum(8, np.minimum(16, 12 * (1 + volatility)))
        slow_period = np.maximum(21, np.minimum(32, 26 * (1 + volatility)))
        signal_period = np.maximum(7, np.minimum(12, 9 * (1 + volatility)))
        
        # Calculate MACD components
        fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
        slow_ema = prices.ewm(span=slow_period, adjust=False).mean()
        macd = fast_ema - slow_ema
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        histogram = macd - signal
        
        return macd, signal, histogram

    def visualize_regime_classification(self, prices: pd.Series, high: pd.Series, 
                                      low: pd.Series, volume: pd.Series) -> None:
        """Create interactive visualization of regime classification"""
        # Calculate hybrid indicators
        hybrid_indicators = self.calculate_hybrid_ichimoku_macd(prices, high, low)
        
        # Calculate regime
        regime = self.detect_market_regime(prices)
        
        # Create main price plot with hybrid indicators
        self.fig.add_trace(
            go.Candlestick(
                x=prices.index,
                open=prices,
                high=high,
                low=low,
                close=prices,
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add Ichimoku Cloud
        self.fig.add_trace(
            go.Scatter(
                x=prices.index,
                y=hybrid_indicators['senkou_span_a'],
                fill=None,
                line=dict(color='rgba(76,175,80,0.3)'),
                name='Senkou Span A'
            ),
            row=1, col=1
        )
        
        self.fig.add_trace(
            go.Scatter(
                x=prices.index,
                y=hybrid_indicators['senkou_span_b'],
                fill='tonexty',
                line=dict(color='rgba(255,82,82,0.3)'),
                name='Senkou Span B'
            ),
            row=1, col=1
        )
        
        # Add hybrid signal heatmap
        self.fig.add_trace(
            go.Heatmap(
                x=prices.index,
                y=['Regime Strength'],
                z=[hybrid_indicators['hybrid_signal']],
                colorscale='RdYlGn',
                showscale=False
            ),
            row=2, col=1
        )
        
        # Add volume profile
        self.fig.add_trace(
            go.Bar(
                x=prices.index,
                y=volume,
                marker_color='rgba(128,128,128,0.5)',
                name='Volume'
            ),
            row=3, col=1
        )
        
        # Add momentum indicators
        momentum_score = self._calculate_momentum_score(
            self.calculate_adaptive_rsi(prices),
            {'medium': hybrid_indicators},
            regime
        )
        
        self.fig.add_trace(
            go.Scatter(
                x=prices.index,
                y=pd.Series(momentum_score).rolling(window=20).mean(),
                line=dict(color='purple'),
                name='Momentum Score'
            ),
            row=3, col=1
        )
        
        # Update layout
        self.fig.update_layout(
            height=1000,
            title_text=f"Market Regime Analysis - {regime.regime_type}",
            showlegend=True,
            xaxis3_title="Date",
            yaxis_title="Price",
            yaxis2_title="Regime",
            yaxis3_title="Volume & Momentum"
        )
        
        # Add regime annotations
        self.fig.add_annotation(
            x=prices.index[-1],
            y=prices.iloc[-1],
            text=f"Volatility: {regime.volatility:.2f}\nTrend Strength: {regime.trend_strength:.2f}",
            showarrow=True,
            arrowhead=1
        )
        
        self.fig.show()

    def calculate_volume_profile(self, prices: pd.Series, volume: pd.Series, 
                               n_bins: int = 50) -> pd.DataFrame:
        """Calculate volume profile for regime analysis"""
        price_bins = pd.qcut(prices, n_bins, duplicates='drop')
        volume_profile = volume.groupby(price_bins).sum()
        
        # Calculate POC (Point of Control)
        poc_price = volume_profile.idxmax().left
        
        # Calculate Value Area
        total_volume = volume_profile.sum()
        volume_threshold = total_volume * 0.68  # 68% of total volume
        
        sorted_profile = volume_profile.sort_values(ascending=False)
        cumsum_volume = sorted_profile.cumsum()
        value_area_mask = cumsum_volume <= volume_threshold
        
        value_area_high = sorted_profile[value_area_mask].index[-1].right
        value_area_low = sorted_profile[value_area_mask].index[0].left
        
        return pd.DataFrame({
            'volume': volume_profile,
            'poc': poc_price,
            'va_high': value_area_high,
            'va_low': value_area_low
        })

    def calculate_whale_adjusted_signals(self, prices: pd.Series, whale_data: Dict, 
                                       sentiment_data: Dict) -> Dict[str, float]:
        """Calculate whale-adjusted technical signals"""
        # Calculate base signals
        base_signals = self.generate_adaptive_signals(prices)
        
        # Extract whale metrics
        whale_pressure = self._calculate_whale_pressure(whale_data)
        whale_accumulation = self._calculate_whale_accumulation_zones(prices, whale_data)
        sentiment_score = self._calculate_weighted_sentiment(sentiment_data)
        
        # Calculate whale-sentiment impact score
        impact_score = (
            0.5 * whale_pressure['net_flow'] +
            0.3 * whale_accumulation['zone_score'] +
            0.2 * sentiment_score['composite_score']
        )
        
        # Adjust technical signals
        adjusted_signals = {}
        for signal_type, signal_value in base_signals.items():
            # Apply non-linear transformation based on whale impact
            whale_multiplier = np.tanh(impact_score * 2)  # Non-linear scaling
            adjusted_value = signal_value * (1 + whale_multiplier)
            
            # Apply sentiment-based volatility adjustment
            vol_adjustment = 1 + (sentiment_score['volatility_impact'] * 0.3)
            adjusted_signals[signal_type] = np.clip(adjusted_value * vol_adjustment, -1, 1)
        
        return adjusted_signals

    def _calculate_whale_pressure(self, whale_data: Dict) -> Dict[str, float]:
        """Calculate sophisticated whale pressure metrics"""
        # Calculate net flow from whale transactions
        inflow = pd.Series(whale_data['buys'])
        outflow = pd.Series(whale_data['sells'])
        
        # Calculate exponentially weighted flow
        decay_factor = 0.94
        weighted_inflow = inflow.ewm(alpha=1-decay_factor).mean()
        weighted_outflow = outflow.ewm(alpha=1-decay_factor).mean()
        
        # Calculate flow momentum
        flow_momentum = (weighted_inflow - weighted_outflow).diff()
        
        # Calculate whale concentration
        concentration = self._calculate_whale_concentration(whale_data['addresses'])
        
        # Calculate order book imbalance
        ob_imbalance = self._calculate_orderbook_imbalance(whale_data['orders'])
        
        return {
            'net_flow': (weighted_inflow.iloc[-1] - weighted_outflow.iloc[-1]),
            'flow_momentum': flow_momentum.iloc[-1],
            'concentration': concentration,
            'ob_imbalance': ob_imbalance
        }

    def _calculate_whale_accumulation_zones(self, prices: pd.Series, 
                                      whale_data: Dict) -> Dict[str, float]:
        """Detect and score whale accumulation zones"""
        # Calculate volume-weighted average prices for whale transactions
        vwap_levels = self._calculate_whale_vwap(whale_data['transactions'])
        
        # Identify key price levels with clustering
        price_clusters = self._cluster_whale_prices(vwap_levels)
        
        # Calculate zone strength based on volume and time decay
        zone_strength = {}
        current_price = prices.iloc[-1]
        
        for cluster_center, cluster_data in price_clusters.items():
            # Calculate time-weighted volume
            time_weights = np.exp(-0.1 * (datetime.now() - cluster_data['timestamps']).days)
            weighted_volume = (cluster_data['volumes'] * time_weights).sum()
            
            # Calculate price distance
            price_distance = abs(current_price - cluster_center)
            distance_decay = np.exp(-2 * price_distance / current_price)
            
            # Calculate zone score
            zone_strength[cluster_center] = weighted_volume * distance_decay
        
        # Calculate aggregate zone score
        total_strength = sum(zone_strength.values())
        normalized_strength = {k: v/total_strength for k, v in zone_strength.items()}
        
        return {
            'zone_score': max(normalized_strength.values()),
            'key_levels': normalized_strength,
            'strongest_zone': max(normalized_strength.items(), key=lambda x: x[1])[0]
        }

    def _calculate_weighted_sentiment(self, sentiment_data: Dict) -> Dict[str, float]:
        """Calculate weighted sentiment metrics with source credibility"""
        # Initialize source weights
        source_weights = {
            'twitter': 0.3,
            'reddit': 0.2,
            'news': 0.5
        }
        
        # Calculate weighted sentiment scores
        weighted_scores = {}
        volatility_impact = 0
        
        for source, data in sentiment_data.items():
            # Calculate base sentiment
            base_score = data['positive_ratio'] - data['negative_ratio']
            
            # Calculate engagement score
            engagement = np.log1p(data['engagement_metrics'])
            engagement_norm = engagement / engagement.max()
            
            # Calculate credibility score
            credibility = self._calculate_source_credibility(data['source_metrics'])
            
            # Calculate final weighted score
            weighted_scores[source] = (
                base_score * 
                source_weights[source] * 
                engagement_norm * 
                credibility
            )
            
            # Calculate sentiment volatility impact
            volatility_impact += data['sentiment_volatility'] * source_weights[source]
        
        # Calculate composite score with non-linear transformation
        composite_score = np.tanh(sum(weighted_scores.values()))
        
        return {
            'composite_score': composite_score,
            'source_scores': weighted_scores,
            'volatility_impact': volatility_impact
        }

    def create_advanced_visualization(self, prices: pd.Series, whale_data: Dict,
                                sentiment_data: Dict) -> None:
        """Create advanced interactive visualization with whale and sentiment data"""
        # Calculate all indicators
        whale_signals = self._calculate_whale_pressure(whale_data)
        sentiment_scores = self._calculate_weighted_sentiment(sentiment_data)
        accumulation_zones = self._calculate_whale_accumulation_zones(prices, whale_data)
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                'Price & Whale Zones',
                'Whale Flow & Sentiment',
                'Technical Indicators',
                'Composite Signals'
            ),
            specs=[[{"secondary_y": True}],
                   [{"secondary_y": True}],
                   [{"secondary_y": True}],
                   [{"secondary_y": False}]]
        )
        
        # Add price candlesticks
        fig.add_trace(
            go.Candlestick(
                x=prices.index,
                open=prices['open'],
                high=prices['high'],
                low=prices['low'],
                close=prices['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add whale accumulation zones
        for level, strength in accumulation_zones['key_levels'].items():
            fig.add_shape(
                type="line",
                x0=prices.index[0],
                x1=prices.index[-1],
                y0=level,
                y1=level,
                line=dict(
                    color="rgba(255, 255, 255, 0.3)",
                    width=strength * 10,
                    dash="dot",
                ),
                row=1, col=1
            )
        
        # Add whale flow indicators
        fig.add_trace(
            go.Scatter(
                x=prices.index,
                y=whale_signals['net_flow'],
                name='Whale Net Flow',
                line=dict(color='blue')
            ),
            row=2, col=1
        )
        
        # Add sentiment overlay
        fig.add_trace(
            go.Scatter(
                x=prices.index,
                y=sentiment_scores['composite_score'],
                name='Sentiment Score',
                line=dict(color='purple')
            ),
            row=2, col=1,
            secondary_y=True
        )
        
        # Update layout with advanced features
        fig.update_layout(
            template='plotly_dark',
            height=1200,
            showlegend=True,
            title={
                'text': 'Advanced Market Analysis Dashboard',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            }
        )
        
        # Add range slider
        fig.update_xaxes(rangeslider_visible=True, row=4, col=1)
        
        # Show the figure
        fig.show()