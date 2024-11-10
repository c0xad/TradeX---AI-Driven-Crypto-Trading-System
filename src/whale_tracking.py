from birdeye import BirdeyeAPI
from typing import Dict, List, Optional, Union
import asyncio
import aiohttp
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass
from collections import defaultdict
import time
import logging
from cachetools import TTLCache, cached
from tenacity import retry, stop_after_attempt, wait_exponential
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import networkx as nx
from scipy import stats
import torch
import torch.nn as nn

@dataclass
class WhaleConfig:
    min_transaction_size: float = 100000  # Min USD value for whale tx
    refresh_interval: int = 60  # Seconds
    cache_ttl: int = 300  # Seconds
    max_retries: int = 3
    batch_size: int = 100
    api_rate_limit: int = 10  # Requests per second

class VolatilityAnalyzer:
    def __init__(self, window_sizes=[5, 15, 30, 60]):
        self.window_sizes = window_sizes
        self.price_history = defaultdict(list)
        self.volatility_cache = TTLCache(maxsize=1000, ttl=300)
        self.garch_models = {}
        
    def calculate_relative_strength_index(self, prices, window=14):
        delta = np.diff(prices)
        gain = (delta > 0) * delta
        loss = (delta < 0) * -delta
        
        avg_gain = np.convolve(gain, np.ones(window)/window, mode='valid')
        avg_loss = np.convolve(loss, np.ones(window)/window, mode='valid')
        
        rs = avg_gain / np.maximum(avg_loss, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    async def analyze_price_volatility(self, token_address: str, price_data: List[float]) -> Dict:
        self.price_history[token_address].extend(price_data)
        
        if len(self.price_history[token_address]) > max(self.window_sizes):
            self.price_history[token_address] = self.price_history[token_address][-max(self.window_sizes):]
        
        volatility_metrics = {}
        prices = np.array(self.price_history[token_address])
        
        # Calculate multi-timeframe volatility
        for window in self.window_sizes:
            if len(prices) >= window:
                returns = np.log(prices[1:] / prices[:-1])
                volatility = np.std(returns[-window:]) * np.sqrt(252)
                volatility_metrics[f'volatility_{window}'] = volatility
        
        # Calculate advanced metrics
        volatility_metrics.update({
            'rsi': self.calculate_relative_strength_index(prices)[-1],
            'kurtosis': stats.kurtosis(prices),
            'skewness': stats.skew(prices),
            'garch_volatility': await self._calculate_garch_volatility(prices),
            'realized_volatility': self._calculate_realized_volatility(prices),
            'implied_volatility_ratio': await self._estimate_implied_volatility_ratio(token_address),
            'volatility_regime': self._detect_volatility_regime(prices),
            'tail_risk_metrics': self._calculate_tail_risk(prices)
        })
        
        return volatility_metrics

    async def _calculate_garch_volatility(self, prices: np.ndarray) -> float:
        returns = np.log(prices[1:] / prices[:-1])
        garch = arch_model(returns, vol='Garch', p=1, q=1)
        model_fit = garch.fit(disp='off')
        forecast = model_fit.forecast(horizon=1)
        return np.sqrt(forecast.variance.values[-1, -1])

    def _calculate_tail_risk(self, prices: np.ndarray) -> Dict:
        returns = np.log(prices[1:] / prices[:-1])
        var_95 = np.percentile(returns, 5)
        es_95 = returns[returns <= var_95].mean()
        
        return {
            'value_at_risk_95': var_95,
            'expected_shortfall_95': es_95,
            'tail_dependence': self._estimate_tail_dependence(returns)
        }

class EnhancedWhaleTracker:
    def __init__(self, api_key: str, config: Optional[WhaleConfig] = None):
        self.config = config or WhaleConfig()
        self.api = BirdeyeAPI(api_key)
        self.session = aiohttp.ClientSession()
        
        # Initialize caches
        self.transaction_cache = TTLCache(maxsize=1000, ttl=self.config.cache_ttl)
        self.whale_patterns_cache = TTLCache(maxsize=500, ttl=self.config.cache_ttl)
        
        # Setup async event loop and queues
        self.event_loop = asyncio.get_event_loop()
        self.transaction_queue = asyncio.Queue(maxsize=1000)
        self.processing_queue = asyncio.Queue(maxsize=1000)
        
        # Initialize rate limiting
        self.rate_limiter = asyncio.Semaphore(self.config.api_rate_limit)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('WhaleTracker')
        
        # Initialize metrics collectors
        self.metrics = defaultdict(lambda: defaultdict(float))
        self.active_whales = set()
        
        # Start background tasks
        self.start_background_tasks()
        
        self.volatility_analyzer = VolatilityAnalyzer()
        self.whale_network = nx.DiGraph()
        self.transaction_patterns = defaultdict(list)
        self.ml_model = self._initialize_ml_model()
    def start_background_tasks(self):
        """Initialize background tasks for continuous monitoring"""
        self.event_loop.create_task(self.fetch_transactions_continuous())
        self.event_loop.create_task(self.process_transactions_continuous())
        self.event_loop.create_task(self.update_metrics_continuous())

    def _initialize_ml_model(self):
        class EnhancedWhalePredictor(nn.Module):
            def __init__(self, input_size: int = 10):
                super().__init__()
                
                # LSTM for sequential pattern recognition
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=128,
                    num_layers=3,
                    dropout=0.3,
                    batch_first=True,
                    bidirectional=True
                )
                
                # Attention mechanism
                self.attention = nn.MultiheadAttention(
                    embed_dim=256,  # bidirectional LSTM output
                    num_heads=8
                )
                
                # Convolutional layers for pattern detection
                self.conv_layers = nn.Sequential(
                    nn.Conv1d(input_size, 64, kernel_size=3),
                    nn.ReLU(),
                    nn.BatchNorm1d(64),
                    nn.Conv1d(64, 32, kernel_size=3),
                    nn.ReLU(),
                    nn.BatchNorm1d(32)
                )
                
                # Fully connected layers
                self.fc_layers = nn.Sequential(
                    nn.Linear(288, 128),  # 256 (LSTM) + 32 (Conv)
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 3)  # Predict price direction, magnitude, and confidence
                )
                
            def forward(self, x, hidden=None):
                # Process sequential data
                lstm_out, (h_n, c_n) = self.lstm(x, hidden)
                
                # Apply attention
                attn_out, _ = self.attention(
                    lstm_out.transpose(0, 1),
                    lstm_out.transpose(0, 1),
                    lstm_out.transpose(0, 1)
                )
                attn_out = attn_out.transpose(0, 1)
                
                # Process with convolutions
                conv_input = x.transpose(1, 2)
                conv_out = self.conv_layers(conv_input)
                conv_out = torch.mean(conv_out, dim=2)
                
                # Combine features
                combined = torch.cat((attn_out[:, -1, :], conv_out), dim=1)
                
                # Final prediction
                output = self.fc_layers(combined)
                
                return output

        return EnhancedWhalePredictor()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def fetch_transactions_continuous(self):
        """Continuously fetch whale transactions from Birdeye API"""
        while True:
            try:
                async with self.rate_limiter:
                    # Fetch large transactions
                    transactions = await self.api.get_transactions(
                        min_value=self.config.min_transaction_size,
                        limit=self.config.batch_size
                    )
                    
                    # Enrich transaction data
                    enriched_transactions = await self.enrich_transaction_data(transactions)
                    
                    # Add to processing queue
                    for tx in enriched_transactions:
                        await self.transaction_queue.put(tx)
                        
                    self.logger.info(f"Fetched {len(enriched_transactions)} whale transactions")
                    
            except Exception as e:
                self.logger.error(f"Error fetching transactions: {str(e)}")
                
            await asyncio.sleep(self.config.refresh_interval)

    async def enrich_transaction_data(self, transactions: List[Dict]) -> List[Dict]:
        """Enrich transaction data with additional metrics"""
        enriched_data = []
        
        for tx in transactions:
            try:
                # Fetch token price data
                token_data = await self.api.get_token_data(tx['token_address'])
                
                # Calculate market impact
                market_impact = await self.calculate_market_impact(
                    tx['token_address'],
                    tx['amount'],
                    token_data['price']
                )
                
                # Get wallet history
                wallet_history = await self.get_wallet_history(tx['wallet_address'])
                
                # Calculate behavioral metrics
                behavior_metrics = self.calculate_behavior_metrics(wallet_history)
                
                enriched_tx = {
                    **tx,
                    'market_impact': market_impact,
                    'token_price': token_data['price'],
                    'token_volume': token_data['volume_24h'],
                    'behavior_score': behavior_metrics['behavior_score'],
                    'trading_pattern': behavior_metrics['pattern'],
                    'risk_level': behavior_metrics['risk_level'],
                    'timestamp': datetime.fromtimestamp(tx['timestamp']),
                    'network_centrality': await self.calculate_network_centrality(tx['wallet_address'])
                }
                
                enriched_data.append(enriched_tx)
                
            except Exception as e:
                self.logger.error(f"Error enriching transaction: {str(e)}")
                
        return enriched_data

    async def calculate_market_impact(self, token_address: str, amount: float, current_price: float) -> Dict:
        """Calculate detailed market impact metrics"""
        async with self.rate_limiter:
            # Fetch order book data
            order_book = await self.api.get_order_book(token_address)
            
            # Calculate slippage
            slippage = self.calculate_slippage(amount, order_book)
            
            # Calculate price impact
            price_impact = self.calculate_price_impact(amount, order_book, current_price)
            
            # Calculate liquidity ratio
            liquidity_ratio = amount / order_book['total_liquidity']
            
            # Calculate market depth impact
            depth_impact = self.calculate_depth_impact(amount, order_book)
            
            return {
                'slippage': slippage,
                'price_impact': price_impact,
                'liquidity_ratio': liquidity_ratio,
                'depth_impact': depth_impact,
                'impact_score': self.calculate_impact_score(
                    slippage, price_impact, liquidity_ratio, depth_impact
                )
            }

    def calculate_behavior_metrics(self, wallet_history: List[Dict]) -> Dict:
        """Calculate complex behavioral metrics for whale wallets"""
        # Calculate trading patterns
        trade_patterns = self.analyze_trading_patterns(wallet_history)
        
        # Calculate volatility metrics
        volatility_metrics = self.calculate_volatility_metrics(wallet_history)
        
        # Calculate timing metrics
        timing_metrics = self.analyze_timing_patterns(wallet_history)
        
        # Calculate network influence
        network_metrics = self.calculate_network_metrics(wallet_history)
        
        # Combine metrics into behavior score
        behavior_score = self.compute_behavior_score(
            trade_patterns,
            volatility_metrics,
            timing_metrics,
            network_metrics
        )
        
        return {
            'behavior_score': behavior_score,
            'pattern': trade_patterns['dominant_pattern'],
            'risk_level': self.assess_risk_level(behavior_score, trade_patterns),
            'metrics': {
                'trade_patterns': trade_patterns,
                'volatility': volatility_metrics,
                'timing': timing_metrics,
                'network': network_metrics
            }
        }

    async def process_transactions_continuous(self):
        """Process transactions from queue continuously"""
        while True:
            try:
                # Get transaction from queue
                tx = await self.transaction_queue.get()
                
                # Process transaction
                processed_data = await self.process_transaction(tx)
                
                # Update metrics
                self.update_metrics(processed_data)
                
                # Check for alerts
                await self.check_alerts(processed_data)
                
                # Mark task as done
                self.transaction_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error processing transaction: {str(e)}")
                
            await asyncio.sleep(0.1)

    async def process_transaction(self, tx: Dict) -> Dict:
        """Process individual transaction with advanced analytics"""
        # Calculate advanced metrics
        impact_metrics = await self.calculate_market_impact(
            tx['token_address'],
            tx['amount'],
            tx['token_price']
        )
        
        # Analyze network effects
        network_metrics = await self.analyze_network_effects(tx)
        
        # Calculate risk metrics
        risk_metrics = self.calculate_risk_metrics(tx, impact_metrics, network_metrics)
        
        # Detect patterns
        patterns = self.detect_transaction_patterns(tx)
        
        # Add volatility analysis
        volatility_metrics = await self.volatility_analyzer.analyze_price_volatility(
            tx['token_address'],
            tx['price_history']
        )
        
        # Detect coordinated activity
        coordination_metrics = await self.detect_coordinated_activity(tx)
        
        # Calculate advanced risk metrics
        risk_metrics = self._calculate_advanced_risk_metrics(
            tx,
            volatility_metrics,
            coordination_metrics
        )
        
        processed_data = {
            **tx,
            'volatility_metrics': volatility_metrics,
            'coordination_metrics': coordination_metrics,
            'risk_metrics': risk_metrics,
            'ml_prediction': self._get_ml_prediction(tx, volatility_metrics, coordination_metrics)
        }
        
        await self._update_alert_system(processed_data)
        
        return processed_data

    def update_metrics(self, processed_data: Dict):
        """Update tracking metrics with processed transaction data"""
        wallet = processed_data['wallet_address']
        token = processed_data['token_address']
        
        # Update wallet metrics
        self.metrics['wallets'][wallet] = {
            'total_volume': self.metrics['wallets'].get(wallet, {}).get('total_volume', 0) + 
                          processed_data['amount'] * processed_data['token_price'],
            'transaction_count': self.metrics['wallets'].get(wallet, {}).get('transaction_count', 0) + 1,
            'average_impact': (
                self.metrics['wallets'].get(wallet, {}).get('average_impact', 0) * 
                self.metrics['wallets'].get(wallet, {}).get('transaction_count', 0) +
                processed_data['impact_metrics']['impact_score']
            ) / (self.metrics['wallets'].get(wallet, {}).get('transaction_count', 0) + 1)
        }
        
        # Update token metrics
        self.metrics['tokens'][token] = {
            'total_volume': self.metrics['tokens'].get(token, {}).get('total_volume', 0) + 
                          processed_data['amount'] * processed_data['token_price'],
            'whale_count': len(set(self.metrics['tokens'].get(token, {}).get('whales', set()) | {wallet}))
        }
        
        # Update global metrics
        self.metrics['global']['total_volume'] += processed_data['amount'] * processed_data['token_price']
        self.metrics['global']['transaction_count'] += 1

    async def check_alerts(self, processed_data: Dict):
        """Check for alert conditions based on processed transaction data"""
        alert_conditions = [
            self.check_impact_alert(processed_data),
            self.check_pattern_alert(processed_data),
            self.check_network_alert(processed_data),
            self.check_risk_alert(processed_data)
        ]
        
        if any(alert_conditions):
            await self.trigger_alert(processed_data, alert_conditions)

    async def detect_coordinated_activity(self, processed_data: Dict) -> Dict:
        # Get recent transactions for clustering
        recent_txs = self._get_recent_transactions(processed_data['token_address'])
        
        # Prepare features for clustering
        features = self._prepare_clustering_features(recent_txs)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=0.3, min_samples=3).fit(scaled_features)
        
        # Analyze clusters
        cluster_analysis = self._analyze_clusters(clustering.labels_, recent_txs)
        
        # Build transaction graph
        self._update_transaction_graph(processed_data)
        
        # Detect suspicious patterns
        suspicious_patterns = self._detect_suspicious_patterns(
            processed_data,
            cluster_analysis
        )
        
        # Calculate correlation metrics
        correlation_metrics = await self._calculate_correlation_metrics(
            processed_data['token_address']
        )
        
        return {
            'cluster_analysis': cluster_analysis,
            'suspicious_patterns': suspicious_patterns,
            'correlation_metrics': correlation_metrics,
            'network_metrics': self._calculate_network_metrics(),
            'risk_score': self._calculate_coordinated_risk_score(
                cluster_analysis,
                suspicious_patterns,
                correlation_metrics
            )
        }

    async def _update_alert_system(self, processed_data: Dict):
        alert_conditions = {
            'high_volatility': processed_data['volatility_metrics']['volatility_15'] > 0.5,
            'coordinated_activity': processed_data['coordination_metrics']['risk_score'] > 0.7,
            'suspicious_pattern': len(processed_data['coordination_metrics']['suspicious_patterns']) > 0,
            'network_centrality': processed_data['coordination_metrics']['network_metrics']['centrality'] > 0.8,
            'risk_level': processed_data['risk_metrics']['aggregate_risk'] > 0.75
        }
        
        if any(alert_conditions.values()):
            await self._trigger_advanced_alert(processed_data, alert_conditions)

class WhaleClusterVisualizer:
    def __init__(self, network_dimension: int = 3):
        self.network_dimension = network_dimension
        self.fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "scatter3d"}, {"type": "heatmap"}],
                   [{"type": "scatter"}, {"type": "histogram2d"}]],
            subplot_titles=(
                'Whale Transaction Network',
                'Temporal Heatmap',
                'Price Impact Flow',
                'Volume-Price Distribution'
            )
        )
        self.color_scale = 'Viridis'
        self.layout_template = 'plotly_dark'
        
    def visualize_whale_clusters(self, whale_network: nx.DiGraph, 
                               price_data: pd.DataFrame,
                               transaction_data: List[Dict]) -> None:
        # Calculate network metrics
        eigenvector_centrality = nx.eigenvector_centrality_numpy(whale_network)
        betweenness_centrality = nx.betweenness_centrality(whale_network)
        clustering_coefficients = nx.clustering(whale_network)
        
        # Create 3D network layout
        pos_3d = nx.spring_layout(whale_network, dim=self.network_dimension)
        
        # Add 3D network visualization
        self._add_3d_network_trace(
            whale_network, 
            pos_3d, 
            eigenvector_centrality,
            betweenness_centrality
        )
        
        # Add temporal heatmap
        self._add_temporal_heatmap(transaction_data, price_data)
        
        # Add price impact flow
        self._add_price_impact_flow(transaction_data, price_data)
        
        # Add volume-price distribution
        self._add_volume_price_distribution(transaction_data)
        
        self._update_layout()
        
    def _add_3d_network_trace(self, network: nx.DiGraph, pos: Dict, 
                             eig_centrality: Dict, bet_centrality: Dict):
        # Create edges
        edge_trace = go.Scatter3d(
            x=[], y=[], z=[],
            line=dict(width=1, color='rgba(136, 136, 136, 0.7)'),
            hoverinfo='none',
            mode='lines'
        )
        
        for edge in network.edges():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            edge_trace['x'] += (x0, x1, None)
            edge_trace['y'] += (y0, y1, None)
            edge_trace['z'] += (z0, z1, None)
        
        # Create nodes
        node_trace = go.Scatter3d(
            x=[pos[node][0] for node in network.nodes()],
            y=[pos[node][1] for node in network.nodes()],
            z=[pos[node][2] for node in network.nodes()],
            mode='markers',
            marker=dict(
                size=10,
                color=[eig_centrality[node] for node in network.nodes()],
                colorscale=self.color_scale,
                opacity=0.8,
                colorbar=dict(title='Eigenvector Centrality'),
                symbol=[
                    'circle' if bet_centrality[node] < 0.5 else 'diamond'
                    for node in network.nodes()
                ]
            ),
            text=[f"Centrality: {eig_centrality[node]:.3f}<br>"
                  f"Betweenness: {bet_centrality[node]:.3f}"
                  for node in network.nodes()],
            hoverinfo='text'
        )
        
        self.fig.add_trace(edge_trace, row=1, col=1)
        self.fig.add_trace(node_trace, row=1, col=1)
