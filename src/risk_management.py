import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import deque
from scipy.optimize import minimize, LinearConstraint
from scipy.stats import norm, t, skew, kurtosis
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
import pandas as pd
from dataclasses import dataclass
from collections import defaultdict
import cvxopt
from arch import arch_model
from statsmodels.stats.moment_helpers import cov2corr
from empyrical import sortino_ratio, omega_ratio, downside_risk
import networkx as nx
from plotly import make_subplots
from plotly import graph_objects as go
from datetime import datetime

@dataclass
class PositionInfo:
    entry_price: float
    size: float
    stop_loss: float
    trailing_stop: float
    risk_threshold: float
    volatility: float
    var: float
    cvar: float
    heat_score: float
    leverage: float
    drawdown: float
    regime_state: int
    tail_hedge_ratio: float
    risk_contribution: float

@dataclass
class StressScenario:
    name: str
    volatility_shock: float
    correlation_shock: float
    return_shock: float
    liquidity_shock: float
    duration: int  # Days
    recovery_rate: float

@dataclass
class RiskBudget:
    target_allocation: float
    min_allocation: float
    max_allocation: float
    risk_contribution: float
    stress_adjustment: float

class RiskManager:
    def __init__(self, initial_capital: float, confidence_level: float = 0.95,
                 max_position_size: float = 0.2, correlation_threshold: float = 0.7):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, PositionInfo] = {}
        self.confidence_level = confidence_level
        self.max_position_size = max_position_size
        self.correlation_threshold = correlation_threshold
        self.historical_returns = {}
        self.risk_free_rate = 0.02  # Annual risk-free rate
        self.max_leverage = 3.0
        self.min_leverage = 0.5
        self.drawdown_threshold = 0.15
        self.rolling_window = 252
        self.regime_window = 63
        self.drawdown_history = deque(maxlen=self.rolling_window)
        self.volatility_regimes = GaussianMixture(n_components=3, random_state=42)
        self.risk_clusters = {}
        self.tail_hedge_active = False
        self.pca = PCA(n_components=3)
        self.factor_loadings = {}

    def set_stop_loss(self, symbol, stop_loss_price):
        if symbol in self.positions:
            self.positions[symbol].stop_loss = stop_loss_price

    def apply_trailing_stop(self, symbol, current_price):
        if symbol in self.positions:
            entry_price = self.positions[symbol].entry_price
            trailing_stop_price = entry_price * (1 - self.positions[symbol].trailing_stop_percentage)
            if current_price > trailing_stop_price:
                self.positions[symbol].trailing_stop = current_price * (1 - self.positions[symbol].trailing_stop_percentage)

    def rebalance_trades(self, market_volatility):
        for symbol, position in self.positions.items():
            if market_volatility > position.risk_threshold:
                self.reduce_position(symbol)

    def reduce_position(self, symbol):
        if symbol in self.positions:
            # Logic to reduce the position size
            pass

    def update_capital(self, profit_loss: float):
        self.current_capital += profit_loss
        # Rebalance positions based on new capital
        for symbol in self.positions:
            self.positions[symbol].size = self.calculate_position_size(
                symbol, 
                self.positions[symbol].volatility,
                0.6
            )

    def get_current_capital(self):
        return self.current_capital

    def calculate_position_size(self, symbol: str, volatility: float, win_rate: float) -> float:
        # Kelly Criterion with volatility adjustment
        kelly_fraction = win_rate - ((1 - win_rate) / (self.positions[symbol].risk_threshold / volatility))
        kelly_fraction = min(kelly_fraction, self.max_position_size)
        volatility_scalar = np.exp(-volatility)
        return self.current_capital * kelly_fraction * volatility_scalar

    def calculate_var_cvar(self, symbol: str, returns: np.ndarray, holding_period: int = 1) -> Tuple[float, float]:
        sorted_returns = np.sort(returns)
        var_index = int((1 - self.confidence_level) * len(sorted_returns))
        var = -sorted_returns[var_index] * np.sqrt(holding_period)
        cvar = -np.mean(sorted_returns[:var_index]) * np.sqrt(holding_period)
        return var, cvar

    def update_position_risk(self, symbol: str, price_data: pd.DataFrame, volatility: float):
        returns = np.log(price_data['close'] / price_data['close'].shift(1)).dropna()
        self.historical_returns[symbol] = returns
        
        # Update risk clusters and factor decomposition
        self.update_risk_clusters()
        self.decompose_risk_factors()
        
        # Detect volatility regime
        regime_state = self.detect_volatility_regime(returns.values)
        var, cvar = self.calculate_rolling_var_cvar(symbol, returns.values)
        
        # Calculate dynamic ATR-based stop loss with regime adjustment
        atr = self.calculate_atr(price_data, period=14)
        regime_multiplier = 1 + (regime_state * 0.5)
        new_stop_loss = price_data['close'].iloc[-1] - (regime_multiplier * 2.5 * atr)
        
        # Calculate dynamic leverage
        leverage = self.calculate_dynamic_leverage(symbol, volatility)
        
        # Update position info with new risk metrics
        self.positions[symbol] = PositionInfo(
            entry_price=self.positions[symbol].entry_price,
            size=self.calculate_position_size(symbol, volatility, 0.6),
            stop_loss=new_stop_loss,
            trailing_stop=self.positions[symbol].trailing_stop,
            risk_threshold=self.calculate_dynamic_risk_threshold(volatility),
            volatility=volatility,
            var=var,
            cvar=cvar,
            heat_score=self.calculate_heat_score(symbol, var, cvar, volatility),
            leverage=leverage,
            drawdown=0.0,
            regime_state=regime_state,
            tail_hedge_ratio=0.0,
            risk_contribution=0.0
        )
        
        # Manage drawdown and update hedging
        self.manage_drawdown(symbol, price_data['close'].iloc[-1])

    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        return tr.rolling(period).mean().iloc[-1]

    def calculate_dynamic_risk_threshold(self, volatility: float) -> float:
        base_threshold = 0.02  # 2% base risk threshold
        return base_threshold * (1 + np.log1p(volatility))

    def calculate_heat_score(self, symbol: str, var: float, cvar: float, volatility: float) -> float:
        return (0.4 * var + 0.4 * cvar + 0.2 * volatility) / self.positions[symbol].size

    def optimize_portfolio_allocation(self):
        if len(self.positions) < 2:
            return

        returns_df = pd.DataFrame(self.historical_returns)
        correlation_matrix = returns_df.corr()
        
        # Identify highly correlated pairs
        high_correlation_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i,j]) > self.correlation_threshold:
                    high_correlation_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))

        # Adjust position sizes for correlated pairs
        for sym1, sym2 in high_correlation_pairs:
            total_size = self.positions[sym1].size + self.positions[sym2].size
            heat_score1 = self.positions[sym1].heat_score
            heat_score2 = self.positions[sym2].heat_score
            
            # Reallocate based on heat scores
            total_heat = heat_score1 + heat_score2
            self.positions[sym1].size = total_size * (heat_score2 / total_heat)
            self.positions[sym2].size = total_size * (heat_score1 / total_heat)

    def calculate_sharpe_ratio(self, symbol: str) -> float:
        returns = pd.Series(self.historical_returns[symbol])
        excess_returns = returns - (self.risk_free_rate / 252)  # Daily risk-free rate
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    def get_risk_report(self) -> Dict:
        return {
            'total_capital': self.current_capital,
            'portfolio_var': sum(pos.var * pos.size for pos in self.positions.values()),
            'portfolio_cvar': sum(pos.cvar * pos.size for pos in self.positions.values()),
            'position_metrics': {
                symbol: {
                    'size': pos.size,
                    'heat_score': pos.heat_score,
                    'sharpe_ratio': self.calculate_sharpe_ratio(symbol),
                    'var': pos.var,
                    'cvar': pos.cvar
                } for symbol, pos in self.positions.items()
            }
        }

    def detect_volatility_regime(self, returns: np.ndarray) -> int:
        rolling_vol = pd.Series(returns).rolling(self.regime_window).std().values
        regime_data = rolling_vol.reshape(-1, 1)
        labels = self.volatility_regimes.fit_predict(regime_data)
        return labels[-1]

    def calculate_dynamic_leverage(self, symbol: str, volatility: float) -> float:
        position = self.positions[symbol]
        regime = position.regime_state
        base_leverage = self.max_leverage * np.exp(-volatility)
        
        # Adjust leverage based on drawdown
        drawdown_factor = 1 - (position.drawdown / self.drawdown_threshold)
        risk_adjusted_leverage = base_leverage * drawdown_factor
        
        # Further adjust based on risk clustering
        if symbol in self.risk_clusters:
            cluster_size = len(self.risk_clusters[symbol])
            risk_adjusted_leverage *= (1 / np.sqrt(cluster_size))
        
        return np.clip(risk_adjusted_leverage, self.min_leverage, self.max_leverage)

    def update_risk_clusters(self):
        if len(self.historical_returns) < 2:
            return

        returns_matrix = pd.DataFrame(self.historical_returns).fillna(0)
        correlation = returns_matrix.corr()
        distance_matrix = np.sqrt(2 * (1 - correlation))
        clusters = fcluster(linkage(distance_matrix), t=0.7, criterion='distance')
        
        self.risk_clusters = {}
        for symbol, cluster_id in zip(returns_matrix.columns, clusters):
            if cluster_id not in self.risk_clusters:
                self.risk_clusters[cluster_id] = []
            self.risk_clusters[cluster_id].append(symbol)

    def calculate_rolling_var_cvar(self, symbol: str, returns: np.ndarray) -> Tuple[float, float]:
        rolling_vars = []
        rolling_cvars = []
        
        for i in range(len(returns) - self.rolling_window + 1):
            window = returns[i:i+self.rolling_window]
            var_index = int((1 - self.confidence_level) * len(window))
            sorted_window = np.sort(window)
            var = -sorted_window[var_index]
            cvar = -np.mean(sorted_window[:var_index])
            rolling_vars.append(var)
            rolling_cvars.append(cvar)
        
        # Use exponential weighting for recent observations
        weights = np.exp(np.linspace(-1, 0, len(rolling_vars)))
        weights /= weights.sum()
        
        return (np.average(rolling_vars, weights=weights),
                np.average(rolling_cvars, weights=weights))

    def manage_drawdown(self, symbol: str, current_price: float):
        position = self.positions[symbol]
        entry_price = position.entry_price
        drawdown = (entry_price - current_price) / entry_price
        
        self.drawdown_history.append(drawdown)
        max_drawdown = max(self.drawdown_history)
        position.drawdown = max_drawdown
        
        if max_drawdown > self.drawdown_threshold:
            # Progressive de-risking
            reduction_factor = np.exp(-max_drawdown / self.drawdown_threshold)
            position.size *= reduction_factor
            position.leverage *= reduction_factor
            
            # Activate tail risk hedging
            if not self.tail_hedge_active:
                self.implement_tail_hedge(symbol)

    def implement_tail_hedge(self, symbol: str):
        position = self.positions[symbol]
        vol = position.volatility
        
        # Calculate optimal hedge ratio using option-like payoff structure
        otm_factor = 0.95  # 5% OTM
        time_decay = np.exp(-1/252)  # Daily decay factor
        hedge_ratio = stats.norm.cdf(-vol) * time_decay
        
        position.tail_hedge_ratio = hedge_ratio
        self.tail_hedge_active = True
        
        # Adjust position size to account for hedge cost
        hedge_cost = hedge_ratio * position.size * 0.01  # 1% premium
        position.size -= hedge_cost

    def decompose_risk_factors(self):
        if len(self.historical_returns) < 2:
            return

        returns_matrix = pd.DataFrame(self.historical_returns).fillna(0)
        self.pca.fit(returns_matrix)
        
        # Calculate factor loadings for each position
        loadings = self.pca.components_
        for i, symbol in enumerate(returns_matrix.columns):
            self.factor_loadings[symbol] = {
                f'factor_{j+1}': loading[i]
                for j, loading in enumerate(loadings)
            }
            
            # Update risk contribution based on factor loadings
            position = self.positions[symbol]
            position.risk_contribution = np.sqrt(np.sum(np.square(
                [loading for loading in self.factor_loadings[symbol].values()]
            )))

class EnhancedRiskManager:
    def __init__(self, initial_capital: float, risk_free_rate: float = 0.02):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.positions = {}
        self.risk_budgets = {}
        self.historical_data = defaultdict(pd.DataFrame)
        self.stress_scenarios = self._initialize_stress_scenarios()
        self.risk_metrics = defaultdict(dict)
        
        # Risk limits
        self.max_position_size = 0.25
        self.max_sector_exposure = 0.40
        self.max_drawdown = 0.15
        self.min_liquidity_ratio = 2.0
        
        # Dynamic thresholds
        self.vol_threshold = self._calculate_dynamic_vol_threshold()
        self.correlation_threshold = self._calculate_dynamic_corr_threshold()
        
        # Initialize risk decomposition
        self.factor_exposures = {}
        self.systematic_risk = {}
        self.idiosyncratic_risk = {}

    def _initialize_stress_scenarios(self) -> Dict[str, StressScenario]:
        return {
            'market_crash': StressScenario(
                name='market_crash',
                volatility_shock=2.5,
                correlation_shock=0.3,
                return_shock=-0.15,
                liquidity_shock=0.5,
                duration=5,
                recovery_rate=0.6
            ),
            'liquidity_crisis': StressScenario(
                name='liquidity_crisis',
                volatility_shock=1.8,
                correlation_shock=0.4,
                return_shock=-0.08,
                liquidity_shock=0.7,
                duration=7,
                recovery_rate=0.8
            ),
            'correlation_breakdown': StressScenario(
                name='correlation_breakdown',
                volatility_shock=1.5,
                correlation_shock=-0.5,
                return_shock=-0.05,
                liquidity_shock=0.3,
                duration=3,
                recovery_rate=0.9
            )
        }

    def calculate_position_size(self, symbol: str, returns: pd.Series, 
                              market_data: pd.DataFrame) -> float:
        """Calculate optimal position size using multiple approaches"""
        # Calculate various risk metrics
        vol = returns.std() * np.sqrt(252)
        skewness = skew(returns)
        kurt = kurtosis(returns)
        sortino = self.calculate_sortino_ratio(returns)
        omega = self.calculate_omega_ratio(returns)
        
        # Estimate tail risk using EVT
        tail_risk = self._estimate_tail_risk(returns)
        
        # Calculate dynamic Kelly fraction with adjustments
        kelly_fraction = self._calculate_adjusted_kelly(returns, vol, tail_risk)
        
        # Calculate risk parity allocation
        risk_parity_alloc = self._calculate_risk_parity_allocation(symbol, returns)
        
        # Calculate conditional risk allocation
        cond_risk_alloc = self._calculate_conditional_risk_allocation(
            symbol, returns, market_data
        )
        
        # Combine allocations using adaptive weights
        weights = self._calculate_adaptive_weights(
            vol, skewness, kurt, sortino, omega
        )
        
        final_allocation = (
            weights['kelly'] * kelly_fraction +
            weights['risk_parity'] * risk_parity_alloc +
            weights['cond_risk'] * cond_risk_alloc
        )
        
        # Apply position limits and market impact constraints
        final_allocation = self._apply_position_constraints(
            symbol, final_allocation, market_data
        )
        
        return final_allocation * self.current_capital

    def _calculate_adjusted_kelly(self, returns: pd.Series, 
                                volatility: float, 
                                tail_risk: float) -> float:
        """Calculate Kelly fraction with sophisticated adjustments"""
        win_rate = len(returns[returns > 0]) / len(returns)
        avg_win = returns[returns > 0].mean()
        avg_loss = abs(returns[returns < 0].mean())
        
        # Basic Kelly
        kelly = win_rate - ((1 - win_rate) / (avg_win / avg_loss))
        
        # Apply volatility adjustment
        vol_adj = np.exp(-volatility)
        
        # Apply tail risk penalty
        tail_adj = 1 / (1 + tail_risk)
        
        # Apply market regime adjustment
        regime_adj = self._get_regime_adjustment(returns)
        
        return kelly * vol_adj * tail_adj * regime_adj * 0.5  # Half-Kelly for safety

    def _calculate_risk_parity_allocation(self, symbol: str, 
                                        returns: pd.Series) -> float:
        """Calculate risk parity allocation using optimization"""
        n_assets = len(self.positions) + 1
        returns_matrix = pd.DataFrame(self.historical_data)
        returns_matrix[symbol] = returns
        
        # Calculate covariance matrix
        cov_matrix = returns_matrix.cov().values
        
        def risk_parity_objective(weights):
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            asset_contrib = weights * (np.dot(cov_matrix, weights)) / portfolio_risk
            return np.sum((asset_contrib - portfolio_risk/n_assets)**2)
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'ineq', 'fun': lambda x: x}
        ]
        
        result = minimize(
            risk_parity_objective,
            x0=np.ones(n_assets)/n_assets,
            method='SLSQP',
            constraints=constraints
        )
        
        return result.x[-1]  # Return allocation for the new symbol

    def _calculate_conditional_risk_allocation(self, symbol: str,
                                            returns: pd.Series,
                                            market_data: pd.DataFrame) -> float:
        """Calculate conditional risk allocation based on market state"""
        # Estimate conditional volatility using GARCH
        garch_model = arch_model(
            returns,
            vol='Garch',
            p=1,
            q=1,
            dist='skewt'
        )
        garch_result = garch_model.fit(disp='off')
        cond_vol = garch_result.conditional_volatility[-1]
        
        # Calculate conditional correlations
        returns_matrix = pd.DataFrame(self.historical_data)
        returns_matrix[symbol] = returns
        
        # Use DCC-GARCH for dynamic correlations
        cond_corr = self._estimate_dynamic_correlation(returns_matrix)
        
        # Calculate conditional betas
        market_returns = market_data['market_returns']
        cond_beta = self._calculate_conditional_beta(returns, market_returns)
        
        # Adjust allocation based on conditional metrics
        base_allocation = 1/np.sqrt(len(self.positions) + 1)
        vol_adj = 1/cond_vol
        corr_adj = 1 - np.mean(np.abs(cond_corr[-1]))
        beta_adj = 1/(1 + abs(cond_beta))
        
        return base_allocation * vol_adj * corr_adj * beta_adj

    def run_stress_test(self, scenario: StressScenario) -> Dict[str, float]:
        """Run comprehensive stress test simulation"""
        initial_portfolio_value = self.current_capital
        stressed_positions = self.positions.copy()
        
        # Apply stress shocks
        for symbol, position in stressed_positions.items():
            # Simulate price shock
            stressed_price = position.entry_price * (1 + scenario.return_shock)
            
            # Apply volatility shock
            stressed_vol = position.volatility * scenario.volatility_shock
            
            # Adjust position size for liquidity shock
            stressed_size = position.size * (1 - scenario.liquidity_shock)
            
            # Calculate new risk metrics under stress
            stressed_var = self._calculate_stressed_var(
                position, 
                stressed_vol,
                scenario.correlation_shock
            )
            
            # Update position with stressed values
            stressed_positions[symbol] = position._replace(
                size=stressed_size,
                volatility=stressed_vol,
                var=stressed_var
            )
        
        # Simulate recovery path
        recovery_path = self._simulate_recovery_path(
            stressed_positions,
            scenario.duration,
            scenario.recovery_rate
        )
        
        # Calculate stress metrics
        max_drawdown = min(recovery_path) / initial_portfolio_value - 1
        recovery_time = self._calculate_recovery_time(recovery_path)
        liquidity_impact = self._calculate_liquidity_impact(
            stressed_positions,
            scenario.liquidity_shock
        )
        
        return {
            'max_drawdown': max_drawdown,
            'recovery_time': recovery_time,
            'liquidity_impact': liquidity_impact,
            'var_increase': np.mean([p.var for p in stressed_positions.values()]) /
                          np.mean([p.var for p in self.positions.values()]),
            'portfolio_impact': (recovery_path[-1] - initial_portfolio_value) /
                              initial_portfolio_value
        }

    def calculate_risk_adjusted_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive risk-adjusted return metrics"""
        excess_returns = returns - self.risk_free_rate/252
        
        # Calculate Sortino ratio with custom target return
        sortino = sortino_ratio(
            returns=excess_returns,
            required_return=0.05/252,  # 5% annual target
            period='daily'
        )
        
        # Calculate Omega ratio
        omega = omega_ratio(
            returns=excess_returns,
            risk_free=0,
            required_return=0.05/252
        )
        
        # Calculate custom tail risk measure
        tail_ratio = self._calculate_tail_ratio(returns)
        
        # Calculate conditional drawdown
        cond_drawdown = self._calculate_conditional_drawdown(returns)
        
        return {
            'sortino_ratio': sortino,
            'omega_ratio': omega,
            'tail_ratio': tail_ratio,
            'conditional_drawdown': cond_drawdown,
            'downside_deviation': downside_risk(returns),
            'gain_loss_ratio': self._calculate_gain_loss_ratio(returns)
        }

    def _estimate_tail_risk(self, returns: pd.Series) -> float:
        """Estimate tail risk using Extreme Value Theory"""
        threshold = np.percentile(returns, 5)
        tail_returns = returns[returns <= threshold]
        
        if len(tail_returns) < 20:
            return np.std(returns) * 2.5
            
        # Fit Generalized Pareto Distribution
        tail_mean = np.mean(tail_returns)
        tail_var = np.var(tail_returns)
        
        # Calculate shape and scale parameters
        xi = 0.5 * (1 + (tail_mean**2 / tail_var))
        beta = tail_mean * (1 + xi)
        
        # Calculate Expected Shortfall
        q = 0.01  # 1% probability
        var = threshold - (beta/xi) * (((len(returns)*q/len(tail_returns))**-xi) - 1)
        es = var/(1-xi) - beta/xi
        
        return -es

    def _calculate_gain_loss_ratio(self, returns: pd.Series) -> float:
        """Calculate asymmetric gain/loss ratio with volume weighting"""
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) == 0:
            return float('inf')
            
        gain_ratio = np.mean(positive_returns) * len(positive_returns)
        loss_ratio = abs(np.mean(negative_returns)) * len(negative_returns)
        
        return gain_ratio / loss_ratio

    def _calculate_conditional_drawdown(self, returns: pd.Series) -> float:
        """Calculate conditional drawdown at risk"""
        cumulative_returns = (1 + returns).cumprod()
        drawdowns = 1 - cumulative_returns/cumulative_returns.cummax()
        sorted_drawdowns = np.sort(drawdowns)
        
        # Calculate 95% conditional drawdown
        threshold_index = int(len(drawdowns) * 0.95)
        return np.mean(sorted_drawdowns[threshold_index:])

    def _calculate_adaptive_weights(self, volatility: float, skewness: float,
                                  kurtosis: float, sortino: float,
                                  omega: float) -> Dict[str, float]:
        """Calculate adaptive weights for different allocation methods"""
        # Base weights
        weights = {
            'kelly': 0.4,
            'risk_parity': 0.3,
            'cond_risk': 0.3
        }
        
        # Adjust for market conditions
        if abs(skewness) > 1:  # High skewness
            weights['kelly'] *= 0.8
            weights['risk_parity'] *= 1.1
            weights['cond_risk'] *= 1.1
            
        if kurtosis > 5:  # Fat tails
            weights['kelly'] *= 0.7
            weights['risk_parity'] *= 1.2
            weights['cond_risk'] *= 1.1
            
        if volatility > self.vol_threshold:  # High volatility
            weights['kelly'] *= 0.6
            weights['risk_parity'] *= 1.3
            weights['cond_risk'] *= 1.1
            
        # Normalize weights
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}

class AdvancedRiskAnalytics:
    def __init__(self, config: Dict):
        self.config = config
        self.monte_carlo_sims = 10000
        self.confidence_levels = [0.99, 0.975, 0.95, 0.90]
        self.historical_scenarios = self._initialize_historical_scenarios()
        self.risk_dashboard = RiskVisualizationDashboard()
        
        # Initialize risk metrics storage
        self.risk_metrics = {
            'var_metrics': defaultdict(list),
            'tail_events': defaultdict(list),
            'drawdown_paths': defaultdict(list),
            'whale_impacts': defaultdict(list),
            'correlation_matrices': deque(maxlen=100)
        }
        
        # Setup stress test parameters
        self.stress_params = {
            'volatility_shock': [1.5, 2.0, 2.5, 3.0],
            'correlation_shock': [0.2, 0.4, 0.6, 0.8],
            'liquidity_shock': [0.3, 0.5, 0.7, 0.9]
        }

    async def run_monte_carlo_simulation(self, returns: pd.Series, 
                                       positions: Dict[str, PositionInfo],
                                       whale_data: Dict) -> Dict:
        """Run comprehensive Monte Carlo simulation with whale impact"""
        # Estimate distribution parameters with EVT
        tail_params = self._estimate_evt_parameters(returns)
        
        # Generate scenarios
        scenarios = await self._generate_scenarios(
            returns, tail_params, whale_data
        )
        
        # Calculate portfolio paths
        paths = self._calculate_portfolio_paths(scenarios, positions)
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(paths)
        
        # Update visualization
        await self.risk_dashboard.update_monte_carlo_view(
            paths, risk_metrics
        )
        
        return risk_metrics

    def _estimate_evt_parameters(self, returns: pd.Series) -> Dict:
        """Estimate EVT parameters using Peak-Over-Threshold method"""
        threshold = np.percentile(returns, 5)
        exceedances = returns[returns < threshold]
        
        if len(exceedances) < 20:
            return {'xi': 0.5, 'beta': np.std(returns)}
            
        # Fit Generalized Pareto Distribution
        excess = -(exceedances - threshold)
        params = genpareto.fit(excess)
        
        return {
            'xi': params[0],  # shape parameter
            'beta': params[2],  # scale parameter
            'threshold': threshold
        }

    async def _generate_scenarios(self, returns: pd.Series, 
                                tail_params: Dict,
                                whale_data: Dict) -> np.ndarray:
        """Generate scenarios incorporating whale impact"""
        # Generate base scenarios
        base_scenarios = self._generate_base_scenarios(returns, tail_params)
        
        # Calculate whale impact factors
        whale_impacts = await self._calculate_whale_impacts(whale_data)
        
        # Adjust scenarios for whale activity
        adjusted_scenarios = self._adjust_scenarios_for_whales(
            base_scenarios, whale_impacts
        )
        
        return adjusted_scenarios

    def _generate_base_scenarios(self, returns: pd.Series, 
                               tail_params: Dict) -> np.ndarray:
        """Generate base scenarios using mixture of normal and EVT"""
        n_days = 252
        scenarios = np.zeros((self.monte_carlo_sims, n_days))
        
        # Generate normal scenarios
        normal_scenarios = np.random.normal(
            loc=returns.mean(),
            scale=returns.std(),
            size=(self.monte_carlo_sims, n_days)
        )
        
        # Generate tail scenarios using GPD
        tail_scenarios = genpareto.rvs(
            c=tail_params['xi'],
            loc=tail_params['threshold'],
            scale=tail_params['beta'],
            size=(self.monte_carlo_sims, n_days)
        )
        
        # Combine scenarios based on threshold exceedance
        tail_prob = len(returns[returns < tail_params['threshold']]) / len(returns)
        tail_mask = np.random.random((self.monte_carlo_sims, n_days)) < tail_prob
        
        scenarios = np.where(tail_mask, -tail_scenarios, normal_scenarios)
        return scenarios

    async def _calculate_whale_impacts(self, whale_data: Dict) -> pd.DataFrame:
        """Calculate whale impact factors using network analysis"""
        # Create whale transaction network
        G = nx.DiGraph()
        
        for tx in whale_data['transactions']:
            G.add_edge(
                tx['from_address'],
                tx['to_address'],
                weight=tx['amount'],
                timestamp=tx['timestamp']
            )
        
        # Calculate network metrics
        centrality = nx.eigenvector_centrality_numpy(G, weight='weight')
        clustering = nx.clustering(G, weight='weight')
        
        # Calculate temporal concentration
        temporal_concentration = self._calculate_temporal_concentration(
            whale_data['transactions']
        )
        
        # Combine metrics
        impact_factors = pd.DataFrame({
            'centrality': pd.Series(centrality),
            'clustering': pd.Series(clustering),
            'temporal_concentration': temporal_concentration
        })
        
        return impact_factors

    def _adjust_scenarios_for_whales(self, base_scenarios: np.ndarray,
                                   whale_impacts: pd.DataFrame) -> np.ndarray:
        """Adjust scenarios based on whale activity"""
        # Calculate impact multiplier
        impact_multiplier = self._calculate_impact_multiplier(whale_impacts)
        
        # Apply non-linear transformation for extreme impacts
        impact_adjustment = np.tanh(impact_multiplier * base_scenarios)
        
        # Apply volatility adjustment
        vol_adjustment = np.exp(impact_multiplier) * np.std(base_scenarios)
        
        return base_scenarios * (1 + impact_adjustment) * vol_adjustment

    def _calculate_risk_metrics(self, paths: np.ndarray) -> Dict:
        """Calculate comprehensive risk metrics from simulation paths"""
        # Calculate VaR and ES at different confidence levels
        var_metrics = {
            level: np.percentile(paths.min(axis=1), (1 - level) * 100)
            for level in self.confidence_levels
        }
        
        es_metrics = {
            level: paths.min(axis=1)[paths.min(axis=1) <= var_metrics[level]].mean()
            for level in self.confidence_levels
        }
        
        # Calculate maximum drawdown distribution
        drawdowns = self._calculate_drawdown_paths(paths)
        max_drawdowns = drawdowns.max(axis=1)
        
        # Calculate recovery metrics
        recovery_metrics = self._calculate_recovery_metrics(paths, drawdowns)
        
        # Calculate tail dependence
        tail_dependence = self._calculate_tail_dependence(paths)
        
        return {
            'var': var_metrics,
            'es': es_metrics,
            'max_drawdown': {
                'mean': max_drawdowns.mean(),
                'std': max_drawdowns.std(),
                'worst': max_drawdowns.max(),
                'distribution': max_drawdowns
            },
            'recovery': recovery_metrics,
            'tail_dependence': tail_dependence
        }

class RiskVisualizationDashboard:
    def __init__(self):
        self.fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Monte Carlo Paths',
                'Drawdown Distribution',
                'Risk Metrics Evolution',
                'Whale Impact Network',
                'Tail Risk Heatmap',
                'Recovery Analysis'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "histogram"}],
                [{"type": "scatter"}, {"type": "scatter3d"}],
                [{"type": "heatmap"}, {"type": "scatter"}]
            ]
        )
        
        self._setup_layout()
        self.update_time = datetime.now()

    def _setup_layout(self):
        self.fig.update_layout(
            height=1200,
            width=1600,
            showlegend=True,
            template='plotly_dark',
            title_text="Real-time Risk Analytics Dashboard",
            title_x=0.5,
            title_font=dict(size=24)
        )

    async def update_monte_carlo_view(self, paths: np.ndarray, 
                                    risk_metrics: Dict):
        """Update Monte Carlo visualization"""
        # Plot Monte Carlo paths
        self.fig.add_trace(
            go.Scatter(
                y=paths.mean(axis=0),
                mode='lines',
                name='Mean Path',
                line=dict(color='white', width=2)
            ),
            row=1, col=1
        )
        
        # Add confidence intervals
        for conf_level in [0.95, 0.75, 0.5]:
            lower = np.percentile(paths, (1 - conf_level) * 50, axis=0)
            upper = np.percentile(paths, 50 + conf_level * 50, axis=0)
            
            self.fig.add_trace(
                go.Scatter(
                    y=upper,
                    mode='lines',
                    name=f'{conf_level*100}% Upper',
                    line=dict(width=0),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            self.fig.add_trace(
                go.Scatter(
                    y=lower,
                    mode='lines',
                    name=f'{conf_level*100}% Lower',
                    fill='tonexty',
                    line=dict(width=0)
                ),
                row=1, col=1
            )

        # Update drawdown distribution
        self.fig.add_trace(
            go.Histogram(
                x=risk_metrics['max_drawdown']['distribution'],
                nbinsx=50,
                name='Drawdown Distribution'
            ),
            row=1, col=2
        )
        
        # Update risk metrics evolution
        self.fig.add_trace(
            go.Scatter(
                x=[self.update_time],
                y=[risk_metrics['var'][0.99]],
                mode='lines+markers',
                name='99% VaR'
            ),
            row=2, col=1
        )
        
        # Update whale impact network
        self._update_whale_network(row=2, col=2)
        
        # Update tail risk heatmap
        self._update_tail_risk_heatmap(
            risk_metrics['tail_dependence'],
            row=3, col=1
        )
        
        # Update recovery analysis
        self._update_recovery_analysis(
            risk_metrics['recovery'],
            row=3, col=2
        )
        
        self.fig.show()

    def _update_whale_network(self, row: int, col: int):
        """Update 3D whale impact network visualization"""
        # Implementation details for 3D network visualization
        pass

    def _update_tail_risk_heatmap(self, tail_dependence: np.ndarray,
                                 row: int, col: int):
        """Update tail risk heatmap"""
        self.fig.add_trace(
            go.Heatmap(
                z=tail_dependence,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Tail Dependence')
            ),
            row=row, col=col
        )

    def _update_recovery_analysis(self, recovery_metrics: Dict,
                                row: int, col: int):
        """Update recovery analysis visualization"""
        self.fig.add_trace(
            go.Scatter(
                x=recovery_metrics['times'],
                y=recovery_metrics['probabilities'],
                mode='lines+markers',
                name='Recovery Profile'
            ),
            row=row, col=col
        )