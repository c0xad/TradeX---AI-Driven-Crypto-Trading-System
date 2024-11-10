import asyncio
import aiohttp
import aiofiles
import smtplib
import json
import pickle
import zlib
import base64
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Set, Any
from datetime import datetime, timedelta
from pathlib import Path
import logging
import signal
import websockets
import redis
from prometheus_client import Counter, Gauge, Histogram
import yaml
from dataclasses import dataclass, asdict, field
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import telegram
from scipy.stats import norm, entropy
import networkx as nx
from collections import defaultdict, deque
import importlib
import glob
import os

@dataclass
class AlertConfig:
    email_enabled: bool
    telegram_enabled: bool
    whale_threshold: float
    profit_threshold: float
    loss_threshold: float
    volatility_threshold: float
    watched_addresses: Set[str]
    alert_cooldown: int
    priority_levels: Dict[str, int]

@dataclass
class StateCheckpoint:
    timestamp: datetime
    positions: Dict[str, Any]
    performance_metrics: Dict[str, float]
    active_orders: List[Dict]
    market_state: Dict[str, Any]
    risk_metrics: Dict[str, float]
    version: str = "1.0.0"

@dataclass
class AlertMetrics:
    frequency: int = 0
    impact_score: float = 0.0
    accuracy_score: float = 0.0
    response_time: float = 0.0
    false_positive_rate: float = 0.0
    market_correlation: float = 0.0
    priority_history: List[int] = field(default_factory=list)
    
@dataclass
class AlertVisualization:
    fig: go.Figure = None
    network_graph: nx.Graph = field(default_factory=nx.Graph)
    update_time: datetime = field(default_factory=datetime.now)
    metrics_history: Dict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=1000)))

class EnhancedWhaleMomentumBot:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        
        # Initialize Redis for state management
        self.redis_client = redis.Redis(
            host=self.config['redis']['host'],
            port=self.config['redis']['port'],
            db=self.config['redis']['db']
        )
        
        # Initialize alert system
        self.alert_config = self._initialize_alert_config()
        self.alert_history = {}
        
        # Initialize Telegram bot
        if self.alert_config.telegram_enabled:
            self.telegram_bot = telegram.Bot(token=self.config['telegram']['token'])
        
        # Initialize async session
        self.session = None
        self.websocket_connections = {}
        
        # Initialize component locks
        self.state_lock = asyncio.Lock()
        self.alert_lock = asyncio.Lock()
        
        # Initialize checkpoint management
        self.checkpoint_path = Path(self.config['checkpoints']['path'])
        self.checkpoint_interval = self.config['checkpoints']['interval']
        self.last_checkpoint = None
        
        # Initialize metrics
        self._setup_metrics()
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        # Initialize async event loop
        self.loop = asyncio.get_event_loop()
        self.loop.set_exception_handler(self._handle_async_exceptions)

        # Initialize modules
        self.modules = {}
        self._load_modules()

        # Initialize simulation components
        self.simulation_mode = self.config.get('simulation', {}).get('enabled', False)
        if self.simulation_mode:
            self.simulator = MarketSimulator(self.config['simulation'])

    def _load_modules(self):
        """Dynamically load modules from the 'modules' directory."""
        modules_path = os.path.join(os.path.dirname(__file__), 'modules')
        module_files = glob.glob(os.path.join(modules_path, '*.py'))
        for module_file in module_files:
            module_name = os.path.basename(module_file)[:-3]
            if module_name != '__init__':
                spec = importlib.util.spec_from_file_location(module_name, module_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.modules[module_name] = module
                self.logger.info(f"Module '{module_name}' loaded.")

    async def _execute_modules(self, market_data):
        """Execute logic from dynamically loaded modules."""
        for module_name, module in self.modules.items():
            try:
                await module.run(self, market_data)
            except Exception as e:
                self.logger.error(f"Error in module '{module_name}': {str(e)}")
                await self.send_alert(
                    f"Module Error - {module_name}",
                    f"An error occurred in module '{module_name}': {str(e)}",
                    priority="high"
                )
                # Potential recovery mechanism
                if hasattr(module, 'recover'):
                    await module.recover(self)

    async def _initialize_async_session(self):
        """Initialize aiohttp session with custom SSL context and connection pooling"""
        timeout = aiohttp.ClientTimeout(total=30)
        connector = aiohttp.TCPConnector(
            limit=100,
            ssl=False,
            keepalive_timeout=60
        )
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers=self._get_default_headers()
        )

    def _setup_metrics(self):
        """Setup Prometheus metrics"""
        self.metrics = {
            'state_save_duration': Histogram(
                'bot_state_save_duration_seconds',
                'Time spent saving bot state'
            ),
            'alert_send_duration': Histogram(
                'bot_alert_send_duration_seconds',
                'Time spent sending alerts'
            ),
            'active_connections': Gauge(
                'bot_active_connections',
                'Number of active websocket connections'
            ),
            'checkpoint_size': Gauge(
                'bot_checkpoint_size_bytes',
                'Size of checkpoint data in bytes'
            ),
            'alert_count': Counter(
                'bot_alerts_total',
                'Total number of alerts sent',
                ['type', 'priority']
            )
        }

    async def save_state(self) -> None:
        """Save bot state to Redis and create checkpoint"""
        async with self.state_lock:
            try:
                start_time = datetime.now()
                
                # Prepare state data
                state = StateCheckpoint(
                    timestamp=start_time,
                    positions=self.get_current_positions(),
                    performance_metrics=self.get_performance_metrics(),
                    active_orders=self.get_active_orders(),
                    market_state=self.get_market_state(),
                    risk_metrics=self.get_risk_metrics()
                )
                
                # Compress and encode state data
                state_data = self._compress_state(state)
                
                # Save to Redis with expiration
                await self.redis_client.setex(
                    f"bot_state:{start_time.isoformat()}",
                    self.config['state_retention_hours'] * 3600,
                    state_data
                )
                
                # Create periodic checkpoint
                if (self.last_checkpoint is None or 
                    (start_time - self.last_checkpoint).seconds >= self.checkpoint_interval):
                    await self._create_checkpoint(state)
                    self.last_checkpoint = start_time
                
                # Update metrics
                self.metrics['state_save_duration'].observe(
                    (datetime.now() - start_time).total_seconds()
                )
                
            except Exception as e:
                self.logger.error(f"Failed to save state: {str(e)}")
                await self.send_alert(
                    "State Save Error",
                    f"Failed to save bot state: {str(e)}",
                    priority="high"
                )
                raise

    async def _create_checkpoint(self, state: StateCheckpoint) -> None:
        """Create a compressed checkpoint file"""
        checkpoint_file = self.checkpoint_path / f"checkpoint_{state.timestamp.strftime('%Y%m%d_%H%M%S')}.gz"
        
        async with aiofiles.open(checkpoint_file, 'wb') as f:
            compressed_data = self._compress_state(state)
            await f.write(compressed_data)
            
        # Update metrics
        self.metrics['checkpoint_size'].set(len(compressed_data))
        
        # Cleanup old checkpoints
        await self._cleanup_old_checkpoints()

    def _compress_state(self, state: StateCheckpoint) -> bytes:
        """Compress state data using zlib"""
        state_dict = asdict(state)
        serialized = json.dumps(state_dict, default=str).encode('utf-8')
        compressed = zlib.compress(serialized, level=9)
        return base64.b85encode(compressed)

    async def load_state(self) -> Optional[StateCheckpoint]:
        """Load bot state from most recent checkpoint or Redis"""
        async with self.state_lock:
            try:
                # Try loading from Redis first
                latest_state_key = await self.redis_client.keys("bot_state:*")
                if latest_state_key:
                    latest_key = max(latest_state_key)
                    state_data = await self.redis_client.get(latest_key)
                    if state_data:
                        return self._decompress_state(state_data)
                
                # Fall back to checkpoint file
                checkpoint_files = list(self.checkpoint_path.glob("checkpoint_*.gz"))
                if not checkpoint_files:
                    return None
                
                latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
                async with aiofiles.open(latest_checkpoint, 'rb') as f:
                    checkpoint_data = await f.read()
                    return self._decompress_state(checkpoint_data)
                    
            except Exception as e:
                self.logger.error(f"Failed to load state: {str(e)}")
                await self.send_alert(
                    "State Load Error",
                    f"Failed to load bot state: {str(e)}",
                    priority="high"
                )
                return None

    async def send_alert(self, title: str, message: str, priority: str = "medium") -> None:
        """Send alert through configured channels"""
        async with self.alert_lock:
            try:
                # Check alert cooldown
                current_time = datetime.now()
                alert_key = f"{title}:{message}"
                if alert_key in self.alert_history:
                    if (current_time - self.alert_history[alert_key]).seconds < self.alert_config.alert_cooldown:
                        return
                
                self.alert_history[alert_key] = current_time
                
                # Prepare alert tasks
                alert_tasks = []
                
                if self.alert_config.email_enabled:
                    alert_tasks.append(self._send_email_alert(title, message, priority))
                
                if self.alert_config.telegram_enabled:
                    alert_tasks.append(self._send_telegram_alert(title, message, priority))
                
                # Send alerts concurrently
                start_time = datetime.now()
                await asyncio.gather(*alert_tasks)
                
                # Update metrics
                self.metrics['alert_send_duration'].observe(
                    (datetime.now() - start_time).total_seconds()
                )
                self.metrics['alert_count'].labels(
                    type='combined',
                    priority=priority
                ).inc()
                
            except Exception as e:
                self.logger.error(f"Failed to send alert: {str(e)}")

    async def _send_email_alert(self, title: str, message: str, priority: str) -> None:
        """Send email alert"""
        try:
            msg = MIMEMultipart()
            msg['Subject'] = f"[{priority.upper()}] {title}"
            msg['From'] = self.config['email']['sender']
            msg['To'] = self.config['email']['recipient']
            
            body = MIMEText(message)
            msg.attach(body)
            
            with smtplib.SMTP_SSL(
                self.config['email']['smtp_server'],
                self.config['email']['smtp_port']
            ) as server:
                server.login(
                    self.config['email']['username'],
                    self.config['email']['password']
                )
                server.send_message(msg)
                
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {str(e)}")

    async def _send_telegram_alert(self, title: str, message: str, priority: str) -> None:
        """Send Telegram alert"""
        try:
            alert_text = f"*{priority.upper()}* - {title}\n\n{message}"
            await self.telegram_bot.send_message(
                chat_id=self.config['telegram']['chat_id'],
                text=alert_text,
                parse_mode='Markdown'
            )
        except Exception as e:
            self.logger.error(f"Failed to send Telegram alert: {str(e)}")

    async def _cleanup_old_checkpoints(self) -> None:
        """Clean up old checkpoint files"""
        try:
            retention_days = self.config['checkpoints']['retention_days']
            cutoff_time = datetime.now() - timedelta(days=retention_days)
            
            checkpoint_files = list(self.checkpoint_path.glob("checkpoint_*.gz"))
            for checkpoint_file in checkpoint_files:
                if datetime.fromtimestamp(checkpoint_file.stat().st_mtime) < cutoff_time:
                    checkpoint_file.unlink()
                    
        except Exception as e:
            self.logger.error(f"Failed to cleanup checkpoints: {str(e)}")

    def _handle_async_exceptions(self, loop: asyncio.AbstractEventLoop, context: Dict) -> None:
        """Handle uncaught async exceptions"""
        exception = context.get('exception', context['message'])
        self.logger.error(f"Async exception: {str(exception)}")
        
        # Send alert for critical errors
        asyncio.create_task(
            self.send_alert(
                "Critical Error",
                f"Uncaught async exception: {str(exception)}",
                priority="critical"
            )
        )

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown"""
        signals = (signal.SIGTERM, signal.SIGINT)
        for s in signals:
            self.loop.add_signal_handler(
                s,
                lambda s=s: asyncio.create_task(self._shutdown(s))
            )

    async def _shutdown(self, signal: signal.Signals) -> None:
        """Handle graceful shutdown"""
        self.logger.info(f"Received exit signal {signal.name}...")
        
        # Save final state
        await self.save_state()
        
        # Close websocket connections
        for ws in self.websocket_connections.values():
            await ws.close()
        
        # Close HTTP session
        if self.session:
            await self.session.close()
        
        # Close Redis connection
        await self.redis_client.close()
        
        # Stop the event loop
        self.loop.stop()

    async def run(self) -> None:
        """Main bot execution loop with simulation support and improved exception handling."""
        try:
            # Initialize async session
            await self._initialize_async_session()

            # Load previous state
            state = await self.load_state()
            if state:
                await self.restore_state(state)

            # Start main loop
            while True:
                try:
                    # Process market data
                    if self.simulation_mode:
                        market_data = self.simulator.get_next_data()
                    else:
                        market_data = await self.fetch_market_data()

                    await self.process_market_data(market_data)
                    await self.update_positions(market_data)
                    await self.manage_risk(market_data)

                    # Execute dynamic modules
                    await self._execute_modules(market_data)

                    # Save state periodically
                    await self.save_state()

                    # Sleep for configured interval
                    await asyncio.sleep(self.config['update_interval'])

                except Exception as e:
                    self.logger.exception("Error in main loop")
                    await self.send_alert(
                        "Main Loop Error",
                        f"An exception occurred: {str(e)}",
                        priority="high"
                    )
                    # Potential recovery mechanism
                    await self._attempt_recovery()

        except Exception as e:
            self.logger.exception("Fatal error")
            await self.send_alert(
                "Fatal Error",
                f"Bot crashed: {str(e)}",
                priority="critical"
            )
            raise

    async def _attempt_recovery(self):
        """Attempt to recover from errors during runtime."""
        try:
            self.logger.info("Attempting recovery...")
            # Reload dynamic modules
            self._load_modules()
            # Reinitialize components if necessary
            # ...additional recovery steps...
            self.logger.info("Recovery successful.")
        except Exception as recovery_exception:
            self.logger.error(f"Recovery failed: {str(recovery_exception)}")
            await self.send_alert(
                "Recovery Failed",
                f"Recovery attempt failed: {str(recovery_exception)}",
                priority="critical"
            )
            # Optionally: Clean shutdown
            await self._shutdown(signal=signal.SIGTERM)

    async def fetch_market_data(self):
        """Fetch real-time market data with improved exception handling."""
        try:
            # ...existing code to fetch market data...
            pass
        except aiohttp.ClientError as e:
            self.logger.error(f"Network error while fetching market data: {str(e)}")
            await self.send_alert(
                "Network Error",
                f"Failed to fetch market data: {str(e)}",
                priority="high"
            )
            # Retry logic or switch to a backup data source
            # ...additional code...
        except Exception as e:
            self.logger.exception("Unexpected error fetching market data")
            raise

class EnhancedAlertSystem:
    def __init__(self, config: Dict):
        self.config = config
        self.alert_metrics = defaultdict(AlertMetrics)
        self.visualization = AlertVisualization()
        
        # Initialize metrics tracking
        self.metrics = {
            'alert_latency': Histogram(
                'alert_latency_seconds',
                'Time taken to process and send alerts',
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
            ),
            'alert_priority': Gauge(
                'alert_priority_level',
                'Current alert priority level',
                ['alert_type']
            ),
            'false_positives': Counter(
                'false_positive_alerts',
                'Number of false positive alerts',
                ['alert_type']
            )
        }
        
        # Initialize visualization components
        self._setup_visualization()
        
        # Setup adaptive thresholds
        self.adaptive_thresholds = self._initialize_adaptive_thresholds()
        
        # Initialize alert correlation network
        self.alert_network = nx.DiGraph()
        
        # Setup alert history with temporal features
        self.alert_history = defaultdict(lambda: {
            'timestamps': deque(maxlen=1000),
            'priorities': deque(maxlen=1000),
            'impacts': deque(maxlen=1000),
            'market_states': deque(maxlen=1000)
        })

    def _setup_visualization(self):
        """Setup interactive dashboard for alert visualization"""
        self.visualization.fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Alert Priority Distribution',
                'Alert Impact vs Market Volatility',
                'Alert Response Time Distribution',
                'Alert Correlation Network',
                'False Positive Rate Trend',
                'Alert Type Distribution'
            ),
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "histogram"}, {"type": "scatter3d"}],
                [{"type": "scatter"}, {"type": "pie"}]
            ]
        )
        
        # Initialize layout
        self.visualization.fig.update_layout(
            height=1200,
            width=1600,
            showlegend=True,
            template='plotly_dark',
            title_text="Real-time Alert Analytics Dashboard",
            title_x=0.5,
            title_font=dict(size=24)
        )

    async def process_alert(self, alert_type: str, message: str, 
                          market_data: Dict) -> Tuple[str, float]:
        """Process alert with adaptive prioritization"""
        start_time = datetime.now()
        
        # Calculate base priority
        base_priority = self._calculate_base_priority(alert_type, message)
        
        # Calculate market impact score
        impact_score = await self._calculate_market_impact(
            alert_type, market_data
        )
        
        # Calculate temporal features
        temporal_score = self._calculate_temporal_features(alert_type)
        
        # Calculate correlation score
        correlation_score = self._calculate_correlation_score(
            alert_type, market_data
        )
        
        # Combine scores with adaptive weights
        priority_score = self._combine_priority_scores(
            base_priority,
            impact_score,
            temporal_score,
            correlation_score,
            market_data['volatility']
        )
        
        # Update metrics
        await self._update_alert_metrics(
            alert_type, 
            priority_score,
            start_time,
            impact_score
        )
        
        # Update visualization
        await self._update_visualization(
            alert_type,
            priority_score,
            impact_score,
            market_data
        )
        
        return self._get_priority_level(priority_score), impact_score

    async def _calculate_market_impact(self, alert_type: str, 
                                     market_data: Dict) -> float:
        """Calculate market impact score using advanced metrics"""
        volatility = market_data.get('volatility', 0)
        volume = market_data.get('volume', 0)
        liquidity = market_data.get('liquidity', 0)
        
        # Calculate volatility impact
        vol_impact = norm.cdf(volatility, loc=0, scale=0.1)
        
        # Calculate volume impact
        vol_ratio = volume / market_data.get('avg_volume', volume)
        volume_impact = np.tanh(vol_ratio - 1)
        
        # Calculate liquidity impact
        liq_ratio = liquidity / market_data.get('avg_liquidity', liquidity)
        liquidity_impact = 1 - np.exp(-liq_ratio)
        
        # Combine impacts with adaptive weights
        weights = self._calculate_adaptive_weights(
            volatility, volume_impact, liquidity_impact
        )
        
        impact_score = (
            weights['volatility'] * vol_impact +
            weights['volume'] * volume_impact +
            weights['liquidity'] * liquidity_impact
        )
        
        return np.clip(impact_score, 0, 1)

    def _calculate_temporal_features(self, alert_type: str) -> float:
        """Calculate temporal features for alert prioritization"""
        history = self.alert_history[alert_type]
        
        if not history['timestamps']:
            return 0.5
            
        # Calculate time-based features
        current_time = datetime.now()
        time_deltas = [
            (current_time - t).total_seconds() 
            for t in history['timestamps']
        ]
        
        # Calculate frequency features
        recent_alerts = sum(1 for dt in time_deltas if dt <= 3600)  # Last hour
        frequency_score = np.tanh(recent_alerts / 10)
        
        # Calculate temporal patterns
        if len(time_deltas) >= 2:
            intervals = np.diff(time_deltas)
            regularity = 1 - np.std(intervals) / (np.mean(intervals) + 1e-6)
        else:
            regularity = 0.5
            
        # Calculate priority trend
        if history['priorities']:
            priority_trend = np.mean(np.diff(list(history['priorities'])))
        else:
            priority_trend = 0
            
        # Combine temporal features
        temporal_score = (
            0.4 * frequency_score +
            0.3 * regularity +
            0.3 * (1 + priority_trend)
        )
        
        return np.clip(temporal_score, 0, 1)

    def _calculate_correlation_score(self, alert_type: str, 
                                  market_data: Dict) -> float:
        """Calculate correlation score using network analysis"""
        if alert_type not in self.alert_network:
            return 0.5
            
        # Calculate network centrality
        centrality = nx.eigenvector_centrality_numpy(
            self.alert_network,
            weight='weight'
        )
        
        # Calculate alert correlation
        correlations = []
        for neighbor in self.alert_network[alert_type]:
            edge_data = self.alert_network[alert_type][neighbor]
            correlation = edge_data.get('correlation', 0)
            weight = edge_data.get('weight', 1)
            correlations.append(correlation * weight)
            
        if not correlations:
            return centrality.get(alert_type, 0.5)
            
        # Combine network features
        correlation_score = (
            0.6 * np.mean(correlations) +
            0.4 * centrality.get(alert_type, 0.5)
        )
        
        return np.clip(correlation_score, 0, 1)

    async def _update_visualization(self, alert_type: str, priority: float,
                                  impact: float, market_data: Dict):
        """Update real-time visualization dashboard"""
        # Update priority distribution
        self.visualization.fig.add_trace(
            go.Bar(
                x=[alert_type],
                y=[priority],
                name=f'Priority - {datetime.now().strftime("%H:%M:%S")}',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Update impact vs volatility scatter
        self.visualization.fig.add_trace(
            go.Scatter(
                x=[market_data['volatility']],
                y=[impact],
                mode='markers',
                marker=dict(
                    size=10,
                    color=priority,
                    colorscale='Viridis',
                    showscale=True
                ),
                name=f'Impact - {datetime.now().strftime("%H:%M:%S")}'
            ),
            row=1, col=2
        )
        
        # Update response time distribution
        response_time = self.alert_metrics[alert_type].response_time
        self.visualization.fig.add_trace(
            go.Histogram(
                x=[response_time],
                nbinsx=20,
                name='Response Time'
            ),
            row=2, col=1
        )
        
        # Update correlation network
        pos = nx.spring_layout(self.alert_network, dim=3)
        edge_trace = go.Scatter3d(
            x=[], y=[], z=[],
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        node_trace = go.Scatter3d(
            x=[], y=[], z=[],
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                )
            )
        )
        
        for edge in self.alert_network.edges():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            edge_trace['x'] += (x0, x1, None)
            edge_trace['y'] += (y0, y1, None)
            edge_trace['z'] += (z0, z1, None)
            
        self.visualization.fig.add_trace(edge_trace, row=2, col=2)
        self.visualization.fig.add_trace(node_trace, row=2, col=2)
        
        # Update false positive trend
        self.visualization.fig.add_trace(
            go.Scatter(
                x=[datetime.now()],
                y=[self.alert_metrics[alert_type].false_positive_rate],
                mode='lines+markers',
                name='False Positive Rate'
            ),
            row=3, col=1
        )
        
        # Update alert type distribution
        alert_counts = defaultdict(int)
        for alert in self.alert_history:
            alert_counts[alert] += len(self.alert_history[alert]['timestamps'])
            
        self.visualization.fig.add_trace(
            go.Pie(
                labels=list(alert_counts.keys()),
                values=list(alert_counts.values()),
                name='Alert Distribution'
            ),
            row=3, col=2
        )
        
        # Update layout
        self.visualization.fig.update_layout(
            uirevision=True,
            showlegend=True
        )

    def _combine_priority_scores(self, base_priority: float, impact_score: float,
                               temporal_score: float, correlation_score: float,
                               volatility: float) -> float:
        """Combine priority scores with adaptive weights"""
        # Calculate adaptive weights based on market volatility
        if volatility > 0.5:  # High volatility regime
            weights = {
                'base': 0.2,
                'impact': 0.4,
                'temporal': 0.2,
                'correlation': 0.2
            }
        else:  # Normal regime
            weights = {
                'base': 0.3,
                'impact': 0.3,
                'temporal': 0.2,
                'correlation': 0.2
            }
            
        # Apply non-linear transformation for extreme scores
        combined_score = (
            weights['base'] * base_priority +
            weights['impact'] * impact_score +
            weights['temporal'] * temporal_score +
            weights['correlation'] * correlation_score
        )
        
        # Apply sigmoid transformation for smoothing
        transformed_score = 1 / (1 + np.exp(-5 * (combined_score - 0.5)))
        
        return transformed_score

if __name__ == "__main__":
    config_path = "config.yaml"
    bot = EnhancedWhaleMomentumBot(config_path)
    
    try:
        bot.loop.run_until_complete(bot.run())
    except KeyboardInterrupt:
        pass
    finally:
        bot.loop.close()

class MarketSimulator:
    """Simulate market data for testing under various conditions."""
    def __init__(self, simulation_config: Dict):
        self.config = simulation_config
        self.scenarios = self._load_scenarios()
        self.current_scenario = None
        self.scenario_index = 0

    def _load_scenarios(self) -> List[Dict]:
        """Load simulation scenarios from configuration."""
        scenarios = self.config.get('scenarios', [])
        return scenarios

    def get_next_data(self) -> Dict:
        """Return the next piece of simulated market data."""
        if not self.current_scenario or self.scenario_index >= len(self.current_scenario['data']):
            self._next_scenario()

        data = self.current_scenario['data'][self.scenario_index]
        self.scenario_index += 1
        return data

    def _next_scenario(self):
        """Move to the next scenario in the list."""
        if self.scenarios:
            self.current_scenario = self.scenarios.pop(0)
            self.scenario_index = 0
            print(f"Starting simulation scenario: {self.current_scenario['name']}")
        else:
            raise StopIteration("No more simulation scenarios available.")