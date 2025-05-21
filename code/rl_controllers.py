"""
Reinforcement Learning Controllers for Adaptive Neural Stimulation

This module implements reinforcement learning-based control algorithms
for adaptive neural stimulation parameter adjustment.
"""

import numpy as np
import time
import random
from collections import deque
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


class ActorCriticController(AdaptiveController):
    """Actor-Critic based controller for parameter adjustment"""
    
    def __init__(self, param_ranges, control_interval=1.0, actor_lr=0.01, critic_lr=0.1, discount_factor=0.9):
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
        """
        super().__init__(param_ranges, control_interval)
        
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.discount_factor = discount_factor
        
        # Discretize the state space
        self.state_bins = 5
        self.state_features = ['beta_power_ch0', 'spike_rate_ch0', 'theta_power_ch0']
        self.feature_ranges = {
            'beta_power_ch0': (0, 3),
            'spike_rate_ch0': (0, 50),
            'theta_power_ch0': (0, 2)
        }
        
        # Initialize actor network (policy)
        # In a simple tabular case, this is a table of state -> action probabilities
        self.actor = {}
        
        # Initialize critic network (value function)
        # In a tabular case, this is a table of state -> value
        self.critic = {}
        
        # Initialize action space
        self.actions = self._build_action_space()
        
        # Track previous state for updates
        self.prev_state = None
        self.prev_action = None
    
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
    
    def _get_action_probs(self, state):
        """Get action probabilities for a state"""
        if state not in self.actor:
            # Initialize with uniform probabilities
            self.actor[state] = np.ones(len(self.actions)) / len(self.actions)
            
        return self.actor[state]
    
    def _get_state_value(self, state):
        """Get the value of a state"""
        if state not in self.critic:
            self.critic[state] = 0.0
            
        return self.critic[state]
    
    def _update_actor_critic(self, state, action, reward, next_state):
        """Update the actor and critic networks"""
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
    
    def update(self, neural_state, target_state, features):
        """Update parameters using Actor-Critic method"""
        # Call the parent method to check control interval
        if time.time() - self.last_update_time < self.control_interval:
            return self.current_params
        
        self.last_update_time = time.time()
        
        # Discretize the current state
        current_state = self._discretize_state(features)
        
        # Calculate reward if we have a previous state
        if self.prev_state is not None:
            reward = self.calculate_reward(neural_state, target_state, features)
            
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
        
        return self.current_params


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
