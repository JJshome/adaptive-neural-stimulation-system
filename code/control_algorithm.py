"""
Control Algorithm for Adaptive Neural Stimulation

This module implements various control algorithms for adaptive neural stimulation.
"""

import numpy as np
import json
import os
import pickle
import time
from enum import Enum


class ControlMethod(Enum):
    """Control method enumeration"""
    PID = 0
    Q_LEARNING = 1
    ACTOR_CRITIC = 2
    MPC = 3


class AdaptiveController:
    """Base class for adaptive controllers"""
    
    def __init__(self, param_ranges, control_interval=1.0):
        """
        Initialize the adaptive controller
        
        Parameters:
        -----------
        param_ranges : dict
            Dictionary of parameter ranges {param_name: (min, max, step)}
        control_interval : float
            Control update interval in seconds
        """
        self.param_ranges = param_ranges
        self.control_interval = control_interval
        self.current_params = {}
        self.last_update_time = time.time()
        
        # Initialize parameters to middle of ranges
        for param, (min_val, max_val, step) in param_ranges.items():
            self.current_params[param] = (min_val + max_val) / 2
        
        # Performance metrics
        self.performance_history = []
    
    def update(self, neural_state, target_state, features):
        """
        Update the stimulation parameters based on neural state
        
        Parameters:
        -----------
        neural_state : int or str
            Current neural state
        target_state : int or str
            Target neural state
        features : dict
            Dictionary of extracted features
            
        Returns:
        --------
        params : dict
            Updated stimulation parameters
        """
        # Check if it's time to update
        current_time = time.time()
        if current_time - self.last_update_time < self.control_interval:
            return self.current_params
        
        self.last_update_time = current_time
        
        # This should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement the update method")
    
    def set_parameters(self, params):
        """Set the current parameters"""
        for param, value in params.items():
            if param in self.param_ranges:
                min_val, max_val, step = self.param_ranges[param]
                # Ensure the value is within range and quantized by step
                value = min(max(min_val, value), max_val)
                steps = round((value - min_val) / step)
                self.current_params[param] = min_val + steps * step
    
    def clip_parameters(self, params):
        """Clip parameters to their allowed ranges"""
        clipped = {}
        for param, value in params.items():
            if param in self.param_ranges:
                min_val, max_val, step = self.param_ranges[param]
                value = min(max(min_val, value), max_val)
                steps = round((value - min_val) / step)
                clipped[param] = min_val + steps * step
            else:
                clipped[param] = value
        return clipped
    
    def log_performance(self, performance_metrics):
        """Log performance metrics"""
        self.performance_history.append({
            'time': time.time(),
            'params': self.current_params.copy(),
            'metrics': performance_metrics
        })
    
    def save_state(self, filepath):
        """Save the controller state"""
        state = {
            'params': self.current_params,
            'param_ranges': self.param_ranges,
            'performance_history': self.performance_history,
            'last_update_time': self.last_update_time
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self, filepath):
        """Load the controller state"""
        if not os.path.exists(filepath):
            print(f"State file {filepath} not found")
            return False
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.current_params = state['params']
        self.param_ranges = state['param_ranges']
        self.performance_history = state['performance_history']
        self.last_update_time = state['last_update_time']
        
        return True


class PIDController(AdaptiveController):
    """PID Controller for parameter adjustment"""
    
    def __init__(self, param_ranges, control_interval=1.0, kp=1.0, ki=0.1, kd=0.2):
        """
        Initialize the PID controller
        
        Parameters:
        -----------
        param_ranges : dict
            Dictionary of parameter ranges {param_name: (min, max, step)}
        control_interval : float
            Control update interval in seconds
        kp, ki, kd : float
            PID gains
        """
        super().__init__(param_ranges, control_interval)
        
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        # Initialize error history
        self.prev_error = 0
        self.integral = 0
        
        # Parameter-specific gains
        self.param_gains = {}
        for param in param_ranges:
            self.param_gains[param] = {'kp': kp, 'ki': ki, 'kd': kd}
    
    def calculate_error(self, features, target_features):
        """Calculate the error between current and target features"""
        error = 0.0
        count = 0
        
        for feature, target in target_features.items():
            if feature in features:
                # Normalize by expected range
                feature_range = max(abs(target), 1.0)
                feature_error = (features[feature] - target) / feature_range
                error += feature_error
                count += 1
        
        if count > 0:
            error /= count  # Average error
        
        return error
    
    def update(self, neural_state, target_state, features):
        """Update parameters using PID control"""
        # Call the parent method to check control interval
        if time.time() - self.last_update_time < self.control_interval:
            return self.current_params
        
        self.last_update_time = time.time()
        
        # Define target features based on the target state
        target_features = {
            'beta_power_ch0': 0.5 if target_state == 'normal' else 1.5,
            'spike_rate_ch0': 20 if target_state == 'normal' else 5,
            'theta_power_ch0': 0.8 if target_state == 'normal' else 0.3
        }
        
        # Calculate error
        error = self.calculate_error(features, target_features)
        
        # Update integral and derivative terms
        self.integral += error * self.control_interval
        derivative = (error - self.prev_error) / self.control_interval
        self.prev_error = error
        
        # Anti-windup for integral term
        max_integral = 10.0
        self.integral = max(-max_integral, min(self.integral, max_integral))
        
        # Update each parameter based on the error
        for param, current_value in self.current_params.items():
            min_val, max_val, step = self.param_ranges[param]
            param_range = max_val - min_val
            
            # Get parameter-specific gains
            kp = self.param_gains[param]['kp']
            ki = self.param_gains[param]['ki']
            kd = self.param_gains[param]['kd']
            
            # Calculate the parameter adjustment
            adjustment = (
                kp * error +
                ki * self.integral +
                kd * derivative
            )
            
            # Scale the adjustment by the parameter range
            adjustment *= param_range * 0.1  # Limit to 10% of range per step
            
            # Update the parameter
            new_value = current_value + adjustment
            
            # Enforce parameter limits and step size
            steps = round((new_value - min_val) / step)
            self.current_params[param] = min_val + steps * step
            self.current_params[param] = max(min_val, min(self.current_params[param], max_val))
        
        # Log performance
        self.log_performance({
            'error': error,
            'integral': self.integral,
            'derivative': derivative
        })
        
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
    
    # Create a PID controller
    controller = PIDController(param_ranges, control_interval=1.0)
    
    # Example features
    features = {
        'beta_power_ch0': 1.8,
        'spike_rate_ch0': 12,
        'theta_power_ch0': 0.5
    }
    
    # Update parameters
    new_params = controller.update('damaged', 'normal', features)
    
    print("Updated parameters:")
    for param, value in new_params.items():
        print(f"  {param}: {value}")
