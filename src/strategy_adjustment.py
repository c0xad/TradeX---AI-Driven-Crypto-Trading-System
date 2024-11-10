import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import deque
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Lambda, concatenate
from tensorflow.keras.optimizers import Adam
from scipy.stats import entropy
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random

@dataclass
class PPOConfig:
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    batch_size: int = 64
    epochs: int = 10

@dataclass
class MarketState:
    regime: str
    volatility: float
    trend_strength: float
    momentum: float
    entropy: float

class DualQNetwork:
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        
        # Build primary and target networks
        self.primary_network = self._build_network()
        self.target_network = self._build_network()
        self.target_network.set_weights(self.primary_network.get_weights())
        
        # Adaptive learning parameters
        self.base_learning_rate = 0.001
        self.min_learning_rate = 0.0001
        self.learning_rate_decay = 0.995
        self.current_learning_rate = self.base_learning_rate

    def _build_network(self) -> Model:
        input_layer = Input(shape=(self.state_size,))
        
        # Advantage stream
        advantage = Dense(64, activation='relu')(input_layer)
        advantage = Dense(64, activation='relu')(advantage)
        advantage = Dense(self.action_size)(advantage)
        advantage = Lambda(lambda x: x - tf.reduce_mean(x, axis=1, keepdims=True))(advantage)
        
        # Value stream
        value = Dense(64, activation='relu')(input_layer)
        value = Dense(64, activation='relu')(value)
        value = Dense(1)(value)
        value = Lambda(lambda x: tf.tile(x, [1, self.action_size]))(value)
        
        # Combine streams
        output = concatenate([value, advantage])
        
        model = Model(inputs=input_layer, outputs=output)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.current_learning_rate))
        return model

    def update_learning_rate(self, market_state: MarketState):
        volatility_factor = np.clip(market_state.volatility, 0.5, 2.0)
        entropy_factor = np.clip(market_state.entropy, 0.5, 1.5)
        
        self.current_learning_rate = np.clip(
            self.base_learning_rate * volatility_factor * entropy_factor,
            self.min_learning_rate,
            self.base_learning_rate
        )
        
        # Update optimizer learning rate
        tf.keras.backend.set_value(
            self.primary_network.optimizer.learning_rate,
            self.current_learning_rate
        )

class PPOAgent:
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.config = PPOConfig()
        
        # Initialize actor and critic networks
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        
        # Initialize optimizers with adaptive learning rates
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)
        
        # Experience buffer
        self.buffer = []
        
        # Adaptive parameters
        self.clip_range = self.config.clip_epsilon
        self.value_clip_range = 0.4

    def _build_actor(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size),
            nn.Softmax(dim=-1)
        )

    def _build_critic(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def update(self, states, actions, rewards, next_states, dones):
        # Convert data to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Get old action probabilities and values
        with torch.no_grad():
            old_probs = self.actor(states)
            old_values = self.critic(states)
            
        # PPO update loop
        for _ in range(self.config.epochs):
            # Actor update
            probs = self.actor(states)
            values = self.critic(states)
            
            # Calculate advantages
            next_values = self.critic(next_states)
            advantages = rewards + (1 - dones) * 0.99 * next_values - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Calculate ratios and surrogate losses
            ratio = probs / (old_probs + 1e-8)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
            
            # Calculate actor loss
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Calculate critic loss
            value_pred_clipped = old_values + torch.clamp(
                values - old_values,
                -self.value_clip_range,
                self.value_clip_range
            )
            value_losses = (values - rewards) ** 2
            value_losses_clipped = (value_pred_clipped - rewards) ** 2
            critic_loss = torch.max(value_losses, value_losses_clipped).mean()
            
            # Calculate entropy bonus
            entropy_loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1))
            
            # Total loss
            total_loss = (
                actor_loss +
                self.config.value_coef * critic_loss -
                self.config.entropy_coef * entropy_loss
            )
            
            # Update networks
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                self.config.max_grad_norm
            )
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(),
                self.config.max_grad_norm
            )
            
            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def adapt_hyperparameters(self, market_state: MarketState):
        # Adapt clip range based on market volatility
        self.clip_range = self.config.clip_epsilon * np.clip(
            1.0 / market_state.volatility,
            0.5,
            2.0
        )
        
        # Adapt value coefficient based on trend strength
        self.config.value_coef = 0.5 * (1 + market_state.trend_strength)
        
        # Adapt entropy coefficient based on market regime
        if market_state.regime in ['trending_volatile', 'choppy_volatile']:
            self.config.entropy_coef *= 1.2
        else:
            self.config.entropy_coef = max(0.01, self.config.entropy_coef * 0.9)

class EnsembleRegimeClassifier:
    def __init__(self, n_estimators: int = 5):
        self.n_estimators = n_estimators
        self.classifiers = []
        self.feature_weights = np.ones(n_estimators) / n_estimators
        self.regime_history = deque(maxlen=100)
        
        self._initialize_classifiers()

    def _initialize_classifiers(self):
        # Initialize different types of classifiers
        for _ in range(self.n_estimators):
            classifier = {
                'technical': self._build_technical_classifier(),
                'volatility': self._build_volatility_classifier(),
                'momentum': self._build_momentum_classifier()
            }
            self.classifiers.append(classifier)

    def _build_technical_classifier(self) -> Model:
        model = Sequential([
            Dense(64, activation='relu', input_shape=(10,)),
            Dense(32, activation='relu'),
            Dense(6, activation='softmax')
        ])
        model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy')
        return model

    def _build_volatility_classifier(self) -> Model:
        model = Sequential([
            Dense(32, activation='relu', input_shape=(5,)),
            Dense(32, activation='relu'),
            Dense(6, activation='softmax')
        ])
        model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy')
        return model

    def _build_momentum_classifier(self) -> Model:
        model = Sequential([
            Dense(48, activation='relu', input_shape=(8,)),
            Dense(24, activation='relu'),
            Dense(6, activation='softmax')
        ])
        model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy')
        return model

    def predict_regime(self, market_data: Dict[str, np.ndarray]) -> MarketState:
        predictions = []
        
        for classifier in self.classifiers:
            technical_pred = classifier['technical'].predict(
                market_data['technical_features']
            )
            volatility_pred = classifier['volatility'].predict(
                market_data['volatility_features']
            )
            momentum_pred = classifier['momentum'].predict(
                market_data['momentum_features']
            )
            
            # Combine predictions using feature weights
            combined_pred = (
                technical_pred * 0.4 +
                volatility_pred * 0.3 +
                momentum_pred * 0.3
            )
            predictions.append(combined_pred)
        
        # Ensemble prediction using weighted voting
        final_prediction = np.average(
            predictions,
            weights=self.feature_weights,
            axis=0
        )
        
        # Calculate prediction entropy for confidence measurement
        prediction_entropy = entropy(final_prediction)
        
        # Get regime state
        regime_state = MarketState(
            regime=self._get_regime_label(final_prediction),
            volatility=market_data['volatility_features'][-1],
            trend_strength=market_data['technical_features'][-1],
            momentum=market_data['momentum_features'][-1],
            entropy=prediction_entropy
        )
        
        self.regime_history.append(regime_state)
        self._update_feature_weights()
        
        return regime_state

    def _get_regime_label(self, prediction: np.ndarray) -> str:
        regime_labels = [
            'trending_volatile',
            'choppy_volatile',
            'trending_normal',
            'ranging_low_vol',
            'trending_low_vol',
            'mixed'
        ]
        return regime_labels[np.argmax(prediction)]

    def _update_feature_weights(self):
        if len(self.regime_history) < 50:
            return
            
        # Calculate prediction stability for each classifier
        stability_scores = []
        for i in range(self.n_estimators):
            regime_changes = sum(
                1 for j in range(1, len(self.regime_history))
                if self.regime_history[j].regime != self.regime_history[j-1].regime
            )
            stability_score = 1.0 / (1.0 + regime_changes)
            stability_scores.append(stability_score)
            
        # Update weights based on stability scores
        total_score = sum(stability_scores)
        self.feature_weights = np.array(stability_scores) / total_score

class EnhancedPPOAgent(PPOAgent):
    def __init__(self, state_size: int, action_size: int, population_size: int = 10):
        super().__init__(state_size, action_size)
        
        # Evolution strategy parameters
        self.population_size = population_size
        self.mutation_rate = 0.02
        self.crossover_rate = 0.7
        self.population = self._initialize_population()
        
        # Advanced ensemble components
        self.lstm_policy = self._build_lstm_policy()
        self.transformer_policy = self._build_transformer_policy()
        self.population_policies = []
        
        # Live validation metrics
        self.validation_metrics = {
            'sharpe_ratio': deque(maxlen=100),
            'sortino_ratio': deque(maxlen=100),
            'max_drawdown': deque(maxlen=100),
            'win_rate': deque(maxlen=100)
        }
        
        # Strategy adaptation parameters
        self.adaptation_threshold = 0.15
        self.min_validation_samples = 50
        self.strategy_history = deque(maxlen=1000)

        # Advanced pre-training components
        self.pretrain_epochs = 50
        self.pretrain_batch_size = 1024
        self.historical_buffer = []
        
        # Advanced architectures
        self.attention_policy = self._build_attention_policy()
        self.gru_value_net = self._build_gru_value_network()
        
        # Prioritized experience replay
        self.priority_alpha = 0.6
        self.priority_beta = 0.4
        self.priority_epsilon = 1e-6
        self.priorities = []
        
        # Parallel processing pools
        self.num_workers = 4
        self.data_pool = ThreadPool(self.num_workers)
        
        # Advanced metrics tracking
        self.metrics = {
            'value_loss': deque(maxlen=100),
            'policy_loss': deque(maxlen=100),
            'entropy': deque(maxlen=100),
            'kl_div': deque(maxlen=100),
            'advantage_mean': deque(maxlen=100),
            'learning_rate': deque(maxlen=100)
        }

    def _build_lstm_policy(self) -> nn.Module:
        return nn.Sequential(
            nn.LSTM(
                input_size=self.state_size,
                hidden_size=128,
                num_layers=3,
                dropout=0.2,
                batch_first=True
            ),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size),
            nn.Softmax(dim=-1)
        )

    def _build_transformer_policy(self) -> nn.Module:
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.state_size,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1
        )
        return nn.Sequential(
            nn.TransformerEncoder(encoder_layer, num_layers=3),
            nn.Linear(self.state_size, self.action_size),
            nn.Softmax(dim=-1)
        )

    def _initialize_population(self) -> List[Dict]:
        population = []
        for _ in range(self.population_size):
            individual = {
                'actor_params': self._mutate_network_params(self.actor.state_dict()),
                'critic_params': self._mutate_network_params(self.critic.state_dict()),
                'fitness': 0.0,
                'age': 0
            }
            population.append(individual)
        return population

    def _mutate_network_params(self, params: Dict) -> Dict:
        mutated_params = {}
        for name, param in params.items():
            mutation = torch.randn_like(param) * self.mutation_rate
            mutated_params[name] = param + mutation
        return mutated_params

    def _crossover_networks(self, parent1: Dict, parent2: Dict) -> Dict:
        child_params = {}
        for name in parent1.keys():
            if random.random() < self.crossover_rate:
                # Interpolate between parents
                alpha = random.random()
                child_params[name] = alpha * parent1[name] + (1 - alpha) * parent2[name]
            else:
                # Inherit from stronger parent
                child_params[name] = parent1[name] if parent1['fitness'] > parent2['fitness'] else parent2[name]
        return child_params

    async def evolve_population(self, validation_data: pd.DataFrame):
        # Evaluate population fitness
        for individual in self.population:
            self.actor.load_state_dict(individual['actor_params'])
            self.critic.load_state_dict(individual['critic_params'])
            
            # Calculate fitness using multiple metrics
            sharpe = self._calculate_sharpe_ratio(validation_data)
            sortino = self._calculate_sortino_ratio(validation_data)
            drawdown = self._calculate_max_drawdown(validation_data)
            
            individual['fitness'] = (
                0.4 * sharpe +
                0.3 * sortino +
                0.3 * (1 / (1 + drawdown))
            )
            individual['age'] += 1

        # Sort population by fitness
        self.population.sort(key=lambda x: x['fitness'], reverse=True)

        # Create new generation
        new_population = []
        elite_size = self.population_size // 5

        # Keep elite individuals
        new_population.extend(self.population[:elite_size])

        # Create offspring through crossover and mutation
        while len(new_population) < self.population_size:
            parent1 = random.choice(self.population[:elite_size])
            parent2 = random.choice(self.population)
            
            child_actor = self._crossover_networks(
                parent1['actor_params'],
                parent2['actor_params']
            )
            child_critic = self._crossover_networks(
                parent1['critic_params'],
                parent2['critic_params']
            )
            
            # Apply mutation
            child_actor = self._mutate_network_params(child_actor)
            child_critic = self._mutate_network_params(child_critic)
            
            new_population.append({
                'actor_params': child_actor,
                'critic_params': child_critic,
                'fitness': 0.0,
                'age': 0
            })

        self.population = new_population

    async def update_with_ensemble(self, states, actions, rewards, next_states, dones):
        # Get predictions from all policies
        with torch.no_grad():
            lstm_probs = self.lstm_policy(states)
            transformer_probs = self.transformer_policy(states)
            ppo_probs = self.actor(states)
            
            # Get population predictions
            population_probs = []
            for individual in self.population[:3]:  # Use top 3 individuals
                self.actor.load_state_dict(individual['actor_params'])
                pop_prob = self.actor(states)
                population_probs.append(pop_prob)

        # Combine predictions using adaptive weights
        ensemble_weights = self._calculate_ensemble_weights([
            lstm_probs, transformer_probs, ppo_probs, *population_probs
        ])
        
        combined_probs = torch.zeros_like(ppo_probs)
        for probs, weight in zip([lstm_probs, transformer_probs, ppo_probs, *population_probs], ensemble_weights):
            combined_probs += probs * weight

        # Update networks using combined predictions
        await super().update(states, actions, rewards, next_states, dones)
        
        # Update LSTM and Transformer policies
        self._update_auxiliary_policies(states, combined_probs, rewards)

    def _calculate_ensemble_weights(self, predictions: List[torch.Tensor]) -> np.ndarray:
        # Calculate diversity scores
        diversity_matrix = torch.zeros((len(predictions), len(predictions)))
        for i in range(len(predictions)):
            for j in range(len(predictions)):
                if i != j:
                    diversity_matrix[i,j] = torch.norm(predictions[i] - predictions[j])

        # Calculate prediction stability
        stability_scores = [
            1 / (1 + torch.std(pred, dim=0).mean())
            for pred in predictions
        ]

        # Combine diversity and stability
        weights = []
        for i in range(len(predictions)):
            diversity_score = diversity_matrix[i].mean()
            stability_score = stability_scores[i]
            
            weight = (0.6 * diversity_score + 0.4 * stability_score)
            weights.append(weight)

        # Normalize weights
        weights = np.array(weights)
        return weights / weights.sum()

    def _update_auxiliary_policies(self, states, target_probs, rewards):
        # Update LSTM policy
        lstm_loss = F.kl_div(
            self.lstm_policy(states).log(),
            target_probs,
            reduction='batchmean'
        )
        lstm_loss.backward()
        
        # Update Transformer policy
        transformer_loss = F.kl_div(
            self.transformer_policy(states).log(),
            target_probs,
            reduction='batchmean'
        )
        transformer_loss.backward()

    async def validate_strategy(self, validation_data: pd.DataFrame) -> bool:
        """Validate strategy adjustments in real-time"""
        if len(validation_data) < self.min_validation_samples:
            return True

        # Calculate validation metrics
        sharpe = self._calculate_sharpe_ratio(validation_data)
        sortino = self._calculate_sortino_ratio(validation_data)
        drawdown = self._calculate_max_drawdown(validation_data)
        win_rate = self._calculate_win_rate(validation_data)

        # Update validation metrics
        self.validation_metrics['sharpe_ratio'].append(sharpe)
        self.validation_metrics['sortino_ratio'].append(sortino)
        self.validation_metrics['max_drawdown'].append(drawdown)
        self.validation_metrics['win_rate'].append(win_rate)

        # Calculate improvement scores
        improvement_scores = []
        for metric_name, metric_values in self.validation_metrics.items():
            if len(metric_values) >= 2:
                current = metric_values[-1]
                previous = metric_values[-2]
                
                if metric_name == 'max_drawdown':
                    improvement = (previous - current) / previous
                else:
                    improvement = (current - previous) / abs(previous)
                    
                improvement_scores.append(improvement)

        # Decision based on weighted improvement
        if improvement_scores:
            weighted_improvement = np.average(
                improvement_scores,
                weights=[0.3, 0.3, 0.2, 0.2]
            )
            return weighted_improvement >= -self.adaptation_threshold

        return True

    def _build_attention_policy(self) -> nn.Module:
        return nn.Sequential(
            nn.MultiheadAttention(
                embed_dim=self.state_size,
                num_heads=4,
                dropout=0.1
            ),
            nn.LayerNorm(self.state_size),
            nn.Dropout(0.2),
            nn.Linear(self.state_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.action_size),
            nn.Softmax(dim=-1)
        )

    def _build_gru_value_network(self) -> nn.Module:
        return nn.Sequential(
            nn.GRU(
                input_size=self.state_size,
                hidden_size=256,
                num_layers=3,
                dropout=0.2,
                bidirectional=True,
                batch_first=True
            ),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    async def pretrain_with_historical_data(self, historical_data: pd.DataFrame):
        """Pre-train the model using historical market data"""
        processed_data = await self._preprocess_historical_data(historical_data)
        
        # Create pre-training datasets
        states, actions, rewards = self._create_pretrain_datasets(processed_data)
        
        # Initialize progress tracking
        best_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        print("Starting pre-training phase...")
        
        for epoch in range(self.pretrain_epochs):
            epoch_losses = []
            
            # Create mini-batches using parallel processing
            batches = self.data_pool.map(
                self._create_batch,
                self._get_batch_indices(len(states), self.pretrain_batch_size)
            )
            
            for batch_states, batch_actions, batch_rewards in batches:
                # Convert to tensors
                batch_states = torch.FloatTensor(batch_states)
                batch_actions = torch.LongTensor(batch_actions)
                batch_rewards = torch.FloatTensor(batch_rewards)
                
                # Forward pass through all networks
                policy_outputs = self.actor(batch_states)
                value_outputs = self.critic(batch_states)
                attention_outputs = self.attention_policy(batch_states)
                gru_values = self.gru_value_net(batch_states)
                
                # Calculate complex losses
                policy_loss = self._calculate_policy_loss(
                    policy_outputs,
                    attention_outputs,
                    batch_actions,
                    batch_rewards
                )
                
                value_loss = self._calculate_value_loss(
                    value_outputs,
                    gru_values,
                    batch_rewards
                )
                
                # Additional auxiliary losses
                entropy_loss = self._calculate_entropy_loss(policy_outputs)
                diversity_loss = self._calculate_diversity_loss(
                    policy_outputs,
                    attention_outputs
                )
                
                # Combined loss with adaptive weighting
                total_loss = (
                    0.4 * policy_loss +
                    0.3 * value_loss +
                    0.2 * entropy_loss +
                    0.1 * diversity_loss
                )
                
                # Backpropagation with gradient clipping
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    max_norm=0.5
                )
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(),
                    max_norm=0.5
                )
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                epoch_losses.append(total_loss.item())
            
            # Early stopping check
            avg_loss = np.mean(epoch_losses)
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # Save best model
                self._save_pretrained_model()
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            # Adaptive learning rate adjustment
            if epoch % 5 == 0:
                self._adjust_learning_rate(avg_loss)
            
            print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")

    def _calculate_policy_loss(
        self,
        policy_outputs: torch.Tensor,
        attention_outputs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor
    ) -> torch.Tensor:
        # Calculate policy gradient loss
        pg_loss = -torch.mean(
            torch.sum(
                policy_outputs * actions.float(),
                dim=1
            ) * rewards
        )
        
        # Calculate KL divergence between policy and attention outputs
        kl_div = torch.nn.functional.kl_div(
            policy_outputs.log(),
            attention_outputs,
            reduction='batchmean'
        )
        
        # Combine losses with adaptive weighting
        policy_loss = pg_loss + 0.1 * kl_div
        
        return policy_loss

    def _calculate_value_loss(
        self,
        value_outputs: torch.Tensor,
        gru_values: torch.Tensor,
        rewards: torch.Tensor
    ) -> torch.Tensor:
        # Calculate MSE loss for both value estimates
        value_loss = F.mse_loss(value_outputs, rewards)
        gru_loss = F.mse_loss(gru_values, rewards)
        
        # Combine losses with uncertainty weighting
        uncertainty = torch.abs(value_outputs - gru_values).mean()
        alpha = torch.sigmoid(uncertainty)
        
        combined_loss = alpha * value_loss + (1 - alpha) * gru_loss
        
        return combined_loss

    def _calculate_entropy_loss(self, policy_outputs: torch.Tensor) -> torch.Tensor:
        entropy = -torch.mean(
            torch.sum(
                policy_outputs * torch.log(policy_outputs + 1e-10),
                dim=1
            )
        )
        return -0.01 * entropy  # Encourage exploration

    def _calculate_diversity_loss(
        self,
        policy_outputs: torch.Tensor,
        attention_outputs: torch.Tensor
    ) -> torch.Tensor:
        # Encourage diversity between policy and attention outputs
        cosine_sim = F.cosine_similarity(
            policy_outputs,
            attention_outputs,
            dim=1
        ).mean()
        
        return 0.05 * cosine_sim

    def _adjust_learning_rate(self, current_loss: float):
        """Adaptive learning rate adjustment based on loss trends"""
        if len(self.metrics['value_loss']) > 0:
            prev_loss = self.metrics['value_loss'][-1]
            ratio = current_loss / prev_loss
            
            if ratio > 0.95:
                for param_group in self.actor_optimizer.param_groups:
                    param_group['lr'] *= 0.9
                for param_group in self.critic_optimizer.param_groups:
                    param_group['lr'] *= 0.9
            elif ratio < 0.5:
                for param_group in self.actor_optimizer.param_groups:
                    param_group['lr'] *= 1.1
                for param_group in self.critic_optimizer.param_groups:
                    param_group['lr'] *= 1.1

    async def _preprocess_historical_data(
        self,
        data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process historical data in parallel"""
        # Split data into chunks for parallel processing
        chunk_size = len(data) // self.num_workers
        data_chunks = [
            data[i:i + chunk_size]
            for i in range(0, len(data), chunk_size)
        ]
        
        # Process chunks in parallel
        processed_chunks = await asyncio.gather(*[
            self._process_data_chunk(chunk)
            for chunk in data_chunks
        ])
        
        # Combine processed chunks
        combined_data = {
            'states': np.concatenate([chunk['states'] for chunk in processed_chunks]),
            'actions': np.concatenate([chunk['actions'] for chunk in processed_chunks]),
            'rewards': np.concatenate([chunk['rewards'] for chunk in processed_chunks])
        }
        
        return combined_data