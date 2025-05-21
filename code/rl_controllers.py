"""
Reinforcement Learning Controllers for Adaptive Neural Stimulation

This module implements reinforcement learning-based control algorithms
for adaptive neural stimulation parameter adjustment.
"""

import numpy as np
import time
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from code.control_algorithm import AdaptiveController


class QLearningController(AdaptiveController):
    """Q-Learning based controller for parameter adjustment"""
    
    def __init__(self, param_ranges, control_interval=1.0, learning_rate=0.1, 
                 discount_factor=0.9, exploration_rate=0.2):
        """
        Initialize the Q-Learning controller
        
        Parameters:
        -----------
        param_ranges : dict
            Dictionary of parameter ranges {param_name: (min, max, step)}
        control_interval : float
            Control update interval in seconds
        learning_rate : float
            Learning rate (alpha)
        discount_factor : float
            Discount factor (gamma)
        exploration_rate : float
            Exploration rate (epsilon)
        """
        super().__init__(param_ranges, control_interval)
        
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        
        # Discretize the state space
        self.state_bins = 5  # Number of bins for each feature
        self.state_features = ['beta_power_ch0', 'spike_rate_ch0', 'theta_power_ch0']
        self.feature_ranges = {
            'beta_power_ch0': (0, 3),
            'spike_rate_ch0': (0, 50),
            'theta_power_ch0': (0, 2)
        }
        
        # Discretize the action space
        self.actions = self._build_action_space()
        
        # Initialize Q-table
        self.q_table = {}
        
        # Track previous state and action for updates
        self.prev_state = None
        self.prev_action = None
        
        # Experience replay buffer
        self.experience_buffer = deque(maxlen=1000)
    
    def _build_action_space(self):
        """Build the discrete action space"""
        actions = []
        
        # For each parameter, define increment, decrement, and no change actions
        for param, (min_val, max_val, step) in self.param_ranges.items():
            actions.append((param, step))      # Increment
            actions.append((param, 0))         # No change
            actions.append((param, -step))     # Decrement
        
        return actions
    
    def _discretize_state(self, features):
        """Discretize the continuous state (features) into a discrete state"""
        state = []
        
        for feature in self.state_features:
            if feature in features:
                value = features[feature]
                min_val, max_val = self.feature_ranges.get(feature, (0, 1))
                
                # Clip value to range
                value = max(min_val, min(value, max_val))
                
                # Discretize
                bin_size = (max_val - min_val) / self.state_bins
                if bin_size > 0:
                    bin_idx = min(int((value - min_val) / bin_size), self.state_bins - 1)
                else:
                    bin_idx = 0
                
                state.append(bin_idx)
            else:
                state.append(0)  # Default bin if feature is missing
        
        return tuple(state)
    
    def _update_q_value(self, state, action_idx, reward, next_state):
        """Update Q-value based on the reward and next state"""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(len(self.actions))
        
        # Q-learning update
        current_q = self.q_table[state][action_idx]
        max_next_q = np.max(self.q_table[next_state])
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state][action_idx] = new_q
    
    def calculate_reward(self, neural_state, target_state, features):
        """Calculate the reward based on current and target state"""
        # Define target features based on the target state
        target_features = {
            'beta_power_ch0': 0.5 if target_state == 'normal' else 1.5,
            'spike_rate_ch0': 20 if target_state == 'normal' else 5,
            'theta_power_ch0': 0.8 if target_state == 'normal' else 0.3
        }
        
        # Calculate error for each feature
        errors = []
        for feature, target in target_features.items():
            if feature in features:
                # Calculate normalized error
                feature_range = max(abs(target), 1.0)
                feature_error = abs(features[feature] - target) / feature_range
                errors.append(feature_error)
        
        if not errors:
            return 0
        
        # Total error is the average of feature errors
        total_error = sum(errors) / len(errors)
        
        # Convert error to reward (high error = low reward)
        reward = 1.0 - min(total_error, 1.0)
        
        # Normalize to range [-1, 1] with 0 as baseline
        reward = 2 * reward - 1
        
        return reward
    
    def learn_parameters(self, n_episodes, simulation_func):
        """
        Learn parameters through simulated episodes
        
        Parameters:
        -----------
        n_episodes : int
            Number of episodes to learn from
        simulation_func : callable
            Function that simulates neural response to parameters
            
        Returns:
        --------
        best_params : dict
            Best parameters found during learning
        """
        best_reward = float('-inf')
        best_params = None
        rewards_history = []
        
        for episode in range(n_episodes):
            # Reset episode
            self.prev_state = None
            self.prev_action = None
            
            # Get initial features
            features = simulation_func(self.current_params)
            
            # Start with a random state for exploration
            if np.random.random() < 0.3:  # 30% chance of random reset
                for param in self.current_params:
                    min_val, max_val, step = self.param_ranges[param]
                    steps = np.random.randint(0, int((max_val - min_val) / step) + 1)
                    self.current_params[param] = min_val + steps * step
            
            # Run episode for several steps
            episode_reward = 0
            for step in range(20):  # Fixed number of steps per episode
                # Update parameters
                new_params = self.update('damaged', 'normal', features)
                
                # Simulate neural response
                next_features = simulation_func(new_params)
                
                # Calculate reward
                reward = self.calculate_reward('damaged', 'normal', next_features)
                episode_reward += reward
                
                # Update features for next step
                features = next_features
            
            # Track best parameters
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_params = self.current_params.copy()
            
            rewards_history.append(episode_reward)
            
            # Print progress
            if (episode + 1) % 10 == 0:
                avg_reward = sum(rewards_history[-10:]) / 10
                print(f"Episode {episode + 1}/{n_episodes}, Avg Reward: {avg_reward:.4f}, Best Reward: {best_reward:.4f}")
        
        # Set parameters to best found
        self.set_parameters(best_params)
        
        return best_params
    
    def get_action(self, state_features):
        """
        Get action based on state features
        
        Parameters:
        -----------
        state_features : dict
            Dictionary of state features
            
        Returns:
        --------
        action : tuple
            (parameter_name, change_value)
        """
        # Discretize state
        state = self._discretize_state(state_features)
        
        # Use epsilon-greedy policy
        if np.random.random() < self.exploration_rate:
            # Exploration: random action
            action_idx = np.random.randint(0, len(self.actions))
        else:
            # Exploitation: best action
            if state in self.q_table:
                action_idx = np.argmax(self.q_table[state])
            else:
                action_idx = np.random.randint(0, len(self.actions))
        
        return self.actions[action_idx]
    
    def update(self, neural_state, target_state, features):
        """Update parameters using Q-learning"""
        # Call the parent method to check control interval
        if time.time() - self.last_update_time < self.control_interval:
            return self.current_params
        
        self.last_update_time = time.time()
        
        # Discretize the current state
        current_state = self._discretize_state(features)
        
        # Calculate reward if we have a previous state
        if self.prev_state is not None:
            reward = self.calculate_reward(neural_state, target_state, features)
            
            # Update Q-values
            self._update_q_value(
                self.prev_state, 
                self.prev_action, 
                reward, 
                current_state
            )
            
            # Store experience for replay
            self.experience_buffer.append(
                (self.prev_state, self.prev_action, reward, current_state)
            )
            
            # Log performance
            self.log_performance({
                'state': current_state,
                'action': self.prev_action,
                'reward': reward
            })
            
            # Experience replay (learn from random past experiences)
            self._experience_replay(batch_size=4)
        
        # Choose action using epsilon-greedy strategy
        if np.random.random() < self.exploration_rate:
            # Exploration: choose random action
            action_idx = np.random.randint(0, len(self.actions))
        else:
            # Exploitation: choose best action
            if current_state in self.q_table:
                action_idx = np.argmax(self.q_table[current_state])
            else:
                action_idx = np.random.randint(0, len(self.actions))
        
        # Apply the selected action
        param, change = self.actions[action_idx]
        if param in self.current_params:
            # Update the parameter
            new_value = self.current_params[param] + change
            
            # Clip to allowed range and steps
            min_val, max_val, step = self.param_ranges[param]
            steps = round((new_value - min_val) / step)
            self.current_params[param] = min_val + steps * step
            self.current_params[param] = max(min_val, min(self.current_params[param], max_val))
        
        # Save the current state and action for the next update
        self.prev_state = current_state
        self.prev_action = action_idx
        
        # Gradually reduce exploration rate (anneal epsilon)
        self.exploration_rate = max(0.05, self.exploration_rate * 0.995)
        
        return self.current_params
    
    def _experience_replay(self, batch_size=4):
        """Learn from random past experiences"""
        if len(self.experience_buffer) < batch_size:
            return
        
        # Sample random experiences
        experiences = random.sample(self.experience_buffer, batch_size)
        
        # Learn from each experience
        for state, action, reward, next_state in experiences:
            self._update_q_value(state, action, reward, next_state)


# Neural network for Actor-Critic architecture
class ActorCriticNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(ActorCriticNetwork, self).__init__()
        
        # Shared base
        self.base = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        base_output = self.base(x)
        
        # Calculate action probabilities and state value
        action_probs = self.actor(base_output)
        state_value = self.critic(base_output)
        
        return action_probs, state_value


class ActorCriticController(AdaptiveController):
    """Actor-Critic based controller for parameter adjustment"""
    
    def __init__(self, param_ranges, control_interval=1.0, actor_lr=0.01, critic_lr=0.1, 
                 discount_factor=0.9, use_neural_network=False):
        """
        Initialize the Actor-Critic controller
        
        Parameters:
        -----------
        param_ranges : dict
            Dictionary of parameter ranges {param_name: (min, max, step)}
        control_interval : float
            Control update interval in seconds
        actor_lr : float
            Actor learning rate
        critic_lr : float
            Critic learning rate
        discount_factor : float
            Discount factor (gamma)
        use_neural_network : bool
            Whether to use neural network implementation
        """
        super().__init__(param_ranges, control_interval)
        
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.discount_factor = discount_factor
        self.use_neural_network = use_neural_network
        
        # Discretize the state space
        self.state_bins = 5
        self.state_features = ['beta_power_ch0', 'spike_rate_ch0', 'theta_power_ch0']
        self.feature_ranges = {
            'beta_power_ch0': (0, 3),
            'spike_rate_ch0': (0, 50),
            'theta_power_ch0': (0, 2)
        }
        
        # Initialize action space
        self.actions = self._build_action_space()
        
        # Track previous state for updates
        self.prev_state = None
        self.prev_action = None
        self.prev_features = None
        
        # Experience buffer
        self.experience_buffer = deque(maxlen=1000)
        
        if use_neural_network:
            # Initialize neural network model
            state_size = len(self.state_features)
            action_size = len(self.actions)
            self.network = ActorCriticNetwork(state_size, action_size)
            
            # Initialize optimizer
            self.optimizer = optim.Adam([
                {'params': self.network.base.parameters()},
                {'params': self.network.actor.parameters(), 'lr': actor_lr},
                {'params': self.network.critic.parameters(), 'lr': critic_lr}
            ])
        else:
            # Initialize actor network (policy)
            # In a simple tabular case, this is a table of state -> action probabilities
            self.actor = {}
            
            # Initialize critic network (value function)
            # In a tabular case, this is a table of state -> value
            self.critic = {}
    
    def _build_action_space(self):
        """Build the discrete action space"""
        actions = []
        
        # For each parameter, define increment, decrement, and no change actions
        for param, (min_val, max_val, step) in self.param_ranges.items():
            actions.append((param, step))      # Increment
            actions.append((param, 0))         # No change
            actions.append((param, -step))     # Decrement
        
        return actions
    
    def _discretize_state(self, features):
        """Discretize the continuous state (features) into a discrete state"""
        state = []
        
        for feature in self.state_features:
            if feature in features:
                value = features[feature]
                min_val, max_val = self.feature_ranges.get(feature, (0, 1))
                
                # Clip value to range
                value = max(min_val, min(value, max_val))
                
                # Discretize
                bin_size = (max_val - min_val) / self.state_bins
                if bin_size > 0:
                    bin_idx = min(int((value - min_val) / bin_size), self.state_bins - 1)
                else:
                    bin_idx = 0
                
                state.append(bin_idx)
            else:
                state.append(0)  # Default bin if feature is missing
        
        return tuple(state)
    
    def _get_state_tensor(self, features):
        """Convert features to normalized state tensor"""
        state = []
        
        for feature in self.state_features:
            if feature in features:
                value = features[feature]
                min_val, max_val = self.feature_ranges.get(feature, (0, 1))
                
                # Clip and normalize to [0, 1]
                normalized_value = (max(min_val, min(value, max_val)) - min_val) / (max_val - min_val)
                state.append(normalized_value)
            else:
                state.append(0.0)  # Default value if feature is missing
        
        return torch.FloatTensor(state)
    
    def _get_action_probs(self, state):
        """Get action probabilities for a state"""
        if isinstance(state, tuple):  # Discrete state
            if state not in self.actor:
                # Initialize with uniform probabilities
                self.actor[state] = np.ones(len(self.actions)) / len(self.actions)
                
            return self.actor[state]
        else:  # Continuous state tensor
            state_tensor = state.unsqueeze(0)  # Add batch dimension
            action_probs, _ = self.network(state_tensor)
            return action_probs.squeeze(0).detach().numpy()
    
    def _get_state_value(self, state):
        """Get the value of a state"""
        if isinstance(state, tuple):  # Discrete state
            if state not in self.critic:
                self.critic[state] = 0.0
                
            return self.critic[state]
        else:  # Continuous state tensor
            state_tensor = state.unsqueeze(0)  # Add batch dimension
            _, state_value = self.network(state_tensor)
            return state_value.item()
    
    def _update_actor_critic(self, state, action, reward, next_state):
        """Update the actor and critic networks"""
        if self.use_neural_network:
            # Neural network update
            
            # Prepare data
            if isinstance(state, tuple):
                # Convert discrete state to normalized features
                state_tensor = torch.FloatTensor([s / self.state_bins for s in state])
                next_state_tensor = torch.FloatTensor([s / self.state_bins for s in next_state])
            else:
                state_tensor = state
                next_state_tensor = next_state
            
            # Forward pass for current state
            action_probs, state_value = self.network(state_tensor.unsqueeze(0))
            
            # Forward pass for next state
            with torch.no_grad():
                _, next_state_value = self.network(next_state_tensor.unsqueeze(0))
            
            # Calculate TD error
            td_error = reward + self.discount_factor * next_state_value - state_value
            
            # Calculate losses
            actor_loss = -torch.log(action_probs[0, action]) * td_error.detach()
            critic_loss = td_error.pow(2)
            
            # Combined loss
            loss = actor_loss + critic_loss
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
            # Tabular update
            
            # Get the current state value
            current_value = self._get_state_value(state)
            
            # Get the next state value
            next_value = self._get_state_value(next_state)
            
            # Calculate the TD error
            td_error = reward + self.discount_factor * next_value - current_value
            
            # Update the critic (value function)
            self.critic[state] = current_value + self.critic_lr * td_error
            
            # Update the actor (policy)
            action_probs = self._get_action_probs(state)
            action_probs[action] += self.actor_lr * td_error
            
            # Ensure probabilities sum to 1
            action_probs = np.maximum(action_probs, 0.01)  # Ensure non-zero probability
            action_probs /= np.sum(action_probs)
            
            self.actor[state] = action_probs
    
    def calculate_reward(self, neural_state, target_state, features):
        """Calculate the reward based on current and target state"""
        # Define target features based on the target state
        target_features = {
            'beta_power_ch0': 0.5 if target_state == 'normal' else 1.5,
            'spike_rate_ch0': 20 if target_state == 'normal' else 5,
            'theta_power_ch0': 0.8 if target_state == 'normal' else 0.3
        }
        
        # Calculate error for each feature
        errors = []
        for feature, target in target_features.items():
            if feature in features:
                # Calculate normalized error
                feature_range = max(abs(target), 1.0)
                feature_error = abs(features[feature] - target) / feature_range
                errors.append(feature_error)
        
        if not errors:
            return 0
        
        # Total error is the average of feature errors
        total_error = sum(errors) / len(errors)
        
        # Convert error to reward (high error = low reward)
        reward = 1.0 - min(total_error, 1.0)
        
        # Normalize to range [-1, 1] with 0 as baseline
        reward = 2 * reward - 1
        
        return reward
    
    def learn_parameters(self, n_episodes, simulation_func):
        """
        Learn parameters through simulated episodes
        
        Parameters:
        -----------
        n_episodes : int
            Number of episodes to learn from
        simulation_func : callable
            Function that simulates neural response to parameters
            
        Returns:
        --------
        best_params : dict
            Best parameters found during learning
        """
        best_reward = float('-inf')
        best_params = None
        rewards_history = []
        
        for episode in range(n_episodes):
            # Reset episode
            self.prev_state = None
            self.prev_action = None
            self.prev_features = None
            
            # Get initial features
            features = simulation_func(self.current_params)
            
            # Start with a random state for exploration
            if np.random.random() < 0.3:  # 30% chance of random reset
                for param in self.current_params:
                    min_val, max_val, step = self.param_ranges[param]
                    steps = np.random.randint(0, int((max_val - min_val) / step) + 1)
                    self.current_params[param] = min_val + steps * step
            
            # Run episode for several steps
            episode_reward = 0
            for step in range(20):  # Fixed number of steps per episode
                # Update parameters
                new_params = self.update('damaged', 'normal', features)
                
                # Simulate neural response
                next_features = simulation_func(new_params)
                
                # Calculate reward
                reward = self.calculate_reward('damaged', 'normal', next_features)
                episode_reward += reward
                
                # Update features for next step
                features = next_features
            
            # Track best parameters
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_params = self.current_params.copy()
            
            rewards_history.append(episode_reward)
            
            # Print progress
            if (episode + 1) % 10 == 0:
                avg_reward = sum(rewards_history[-10:]) / 10
                print(f"Episode {episode + 1}/{n_episodes}, Avg Reward: {avg_reward:.4f}, Best Reward: {best_reward:.4f}")
        
        # Set parameters to best found
        self.set_parameters(best_params)
        
        return best_params
    
    def get_action(self, state_features):
        """
        Get action based on state features
        
        Parameters:
        -----------
        state_features : dict
            Dictionary of state features
            
        Returns:
        --------
        action : tuple
            (parameter_name, change_value)
        """
        if self.use_neural_network:
            # Get state tensor
            state_tensor = self._get_state_tensor(state_features)
            
            # Get action probabilities
            action_probs = self._get_action_probs(state_tensor)
            
            # Sample action based on probabilities
            action_idx = np.random.choice(len(self.actions), p=action_probs)
        else:
            # Discretize state
            state = self._discretize_state(state_features)
            
            # Get action probabilities
            action_probs = self._get_action_probs(state)
            
            # Sample action based on probabilities
            action_idx = np.random.choice(len(self.actions), p=action_probs)
        
        return self.actions[action_idx]
    
    def update(self, neural_state, target_state, features):
        """Update parameters using Actor-Critic method"""
        # Call the parent method to check control interval
        if time.time() - self.last_update_time < self.control_interval:
            return self.current_params
        
        self.last_update_time = time.time()
        
        if self.use_neural_network:
            # Get state tensor
            current_state = self._get_state_tensor(features)
        else:
            # Discretize the current state
            current_state = self._discretize_state(features)
        
        # Calculate reward if we have a previous state
        if self.prev_state is not None and self.prev_features is not None:
            reward = self.calculate_reward(neural_state, target_state, features)
            
            # Store experience
            self.experience_buffer.append((
                self.prev_state, 
                self.prev_action, 
                reward, 
                current_state
            ))
            
            # Update actor and critic
            self._update_actor_critic(
                self.prev_state, 
                self.prev_action, 
                reward, 
                current_state
            )
            
            # Log performance
            self.log_performance({
                'state': current_state,
                'action': self.prev_action,
                'reward': reward,
                'value': self._get_state_value(current_state)
            })
            
            # Perform batch updates occasionally
            if len(self.experience_buffer) >= 32 and random.random() < 0.1:
                self._batch_update(batch_size=16)
        
        # Choose action based on the policy
        action_probs = self._get_action_probs(current_state)
        action_idx = np.random.choice(len(self.actions), p=action_probs)
        
        # Apply the selected action
        param, change = self.actions[action_idx]
        if param in self.current_params:
            # Update the parameter
            new_value = self.current_params[param] + change
            
            # Clip to allowed range and steps
            min_val, max_val, step = self.param_ranges[param]
            steps = round((new_value - min_val) / step)
            self.current_params[param] = min_val + steps * step
            self.current_params[param] = max(min_val, min(self.current_params[param], max_val))
        
        # Save the current state and action for the next update
        self.prev_state = current_state
        self.prev_action = action_idx
        self.prev_features = features.copy()
        
        return self.current_params
    
    def _batch_update(self, batch_size=16):
        """Perform batch update from experience buffer"""
        if len(self.experience_buffer) < batch_size:
            return
        
        # Sample batch of experiences
        batch = random.sample(self.experience_buffer, batch_size)
        
        if self.use_neural_network:
            # Neural network batch update
            states = []
            actions = []
            rewards = []
            next_states = []
            
            for state, action, reward, next_state in batch:
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
            
            # Convert to tensors
            state_batch = torch.stack(states)
            action_batch = torch.LongTensor(actions)
            reward_batch = torch.FloatTensor(rewards)
            next_state_batch = torch.stack(next_states)
            
            # Forward pass
            action_probs, state_values = self.network(state_batch)
            _, next_state_values = self.network(next_state_batch)
            
            # Calculate TD errors
            td_errors = reward_batch + self.discount_factor * next_state_values.squeeze() - state_values.squeeze()
            
            # Actor loss
            # Select action probabilities for the actions that were taken
            selected_action_probs = action_probs[torch.arange(batch_size), action_batch]
            actor_loss = -torch.log(selected_action_probs) * td_errors.detach()
            
            # Critic loss
            critic_loss = td_errors.pow(2)
            
            # Total loss
            loss = actor_loss.mean() + critic_loss.mean()
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
            # Tabular batch update
            for state, action, reward, next_state in batch:
                self._update_actor_critic(state, action, reward, next_state)


# Example usage
if __name__ == "__main__":
    # Define parameter ranges
    param_ranges = {
        'frequency': (10, 200, 5),      # Hz (min, max, step)
        'amplitude': (0.5, 5.0, 0.1),   # mA
        'pulse_width': (50, 500, 10),   # µs
        'duty_cycle': (10, 100, 5)      # %
    }
    
    # Create a Q-learning controller
    controller = QLearningController(param_ranges, control_interval=1.0)
    
    # Example features
    features = {
        'beta_power_ch0': 1.8,
        'spike_rate_ch0': 12,
        'theta_power_ch0': 0.5
    }
    
    # Update parameters
    print("Initial parameters:")
    for param, value in controller.current_params.items():
        print(f"  {param}: {value}")
    
    # Simulate a few updates
    print("\nUpdating parameters...")
    for i in range(10):
        new_params = controller.update('damaged', 'normal', features)
        
        # Modify features based on parameter changes (simulating feedback)
        features['beta_power_ch0'] -= 0.1
        features['spike_rate_ch0'] += 1
    
    print("\nFinal parameters:")
    for param, value in new_params.items():
        print(f"  {param}: {value}")
