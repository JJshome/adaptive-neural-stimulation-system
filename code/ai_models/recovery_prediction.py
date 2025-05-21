"""
Recovery Prediction Model for Adaptive Neural Stimulation System

This module implements a neural network-based prediction model for neural recovery based on
biomarkers, stimulation parameters, and patient data. The model leverages multiple input modalities 
to predict recovery trajectories and optimize stimulation parameters.

Key features:
1. Multi-modal input processing (biomarkers, electrophysiology, patient demographics)
2. Temporal modeling of recovery trajectory
3. Uncertainty quantification in predictions
4. Integration with reinforcement learning for parameter optimization

Author: Adaptive Neural Stimulation Team
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
import matplotlib.pyplot as plt
from scipy import signal
import os
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants for model configuration
INPUT_FEATURES = {
    "electrophysiology": 64,      # Number of electrophysiology features
    "biomarkers": 32,             # Number of molecular biomarker features
    "patient_data": 16,           # Number of patient demographic/clinical features
    "stimulation_params": 8,      # Number of stimulation parameter features
}

# Biomarker importance weights based on recent research
BIOMARKER_WEIGHTS = {
    "BDNF": 0.85,                 # Brain-derived neurotrophic factor
    "GDNF": 0.75,                 # Glial cell-derived neurotrophic factor
    "NGF": 0.70,                  # Nerve growth factor
    "cAMP": 0.80,                 # Cyclic adenosine monophosphate
    "IL10": 0.65,                 # Interleukin 10 (anti-inflammatory)
    "TNFa": -0.60,                # Tumor necrosis factor alpha (pro-inflammatory)
    "IL1b": -0.55,                # Interleukin 1 beta (pro-inflammatory)
    "GAP43": 0.85,                # Growth Associated Protein 43
    "VEGF": 0.60,                 # Vascular endothelial growth factor
}

class RecoveryPredictionModel:
    """Neural recovery prediction model for optimizing stimulation parameters.
    
    This model predicts recovery trajectories and outcomes based on patient data,
    biomarkers, electrophysiological recordings, and stimulation parameters.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        model_path: Optional[str] = None,
        use_uncertainty: bool = True,
        use_attention: bool = True,
    ):
        """Initialize the recovery prediction model.
        
        Args:
            config_path: Path to configuration file
            model_path: Path to pre-trained model weights
            use_uncertainty: Whether to enable uncertainty quantification
            use_attention: Whether to use attention mechanisms for feature importance
        """
        self.config = self._load_config(config_path)
        self.use_uncertainty = use_uncertainty
        self.use_attention = use_attention
        self.model = None
        self.history = None
        self.feature_importance = {}
        
        # Initialize and build the model
        self._build_model()
        
        # Load pre-trained weights if provided
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_weights(model_path)
                logger.info(f"Loaded pre-trained model from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model weights: {e}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load model configuration from file or use defaults.
        
        Args:
            config_path: Path to configuration JSON file
            
        Returns:
            Dictionary containing model configuration
        """
        default_config = {
            "lstm_units": 128,
            "dense_units": [256, 128, 64],
            "dropout_rate": 0.3,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "early_stopping_patience": 10,
            "time_steps": 12,  # Number of time steps for temporal prediction
            "prediction_horizon": 4,  # How many time steps to predict into future
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {config_path}")
                # Update default config with loaded values
                for key, value in config.items():
                    default_config[key] = value
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
        
        return default_config
    
    def _build_model(self):
        """Build the neural network architecture for recovery prediction."""
        
        # Input layers
        electro_input = layers.Input(shape=(self.config["time_steps"], INPUT_FEATURES["electrophysiology"]), name="electrophysiology_input")
        biomarkers_input = layers.Input(shape=(INPUT_FEATURES["biomarkers"],), name="biomarkers_input")
        patient_input = layers.Input(shape=(INPUT_FEATURES["patient_data"],), name="patient_data_input")
        stim_params_input = layers.Input(shape=(INPUT_FEATURES["stimulation_params"],), name="stimulation_params_input")
        
        # Process electrophysiology data with LSTM
        lstm_layer = layers.Bidirectional(layers.LSTM(self.config["lstm_units"], return_sequences=self.use_attention))(electro_input)
        
        if self.use_attention:
            # Apply attention mechanism to LSTM outputs
            attention = layers.Dense(1, activation='tanh')(lstm_layer)
            attention = layers.Flatten()(attention)
            attention = layers.Activation('softmax')(attention)
            attention = layers.RepeatVector(2 * self.config["lstm_units"])(attention)
            attention = layers.Permute([2, 1])(attention)
            
            # Apply attention weights to LSTM outputs
            lstm_attention = layers.Multiply()([lstm_layer, attention])
            lstm_attention = layers.Lambda(lambda x: keras.backend.sum(x, axis=1))(lstm_attention)
            electro_features = lstm_attention
        else:
            # Use last LSTM output if not using attention
            electro_features = layers.LSTM(self.config["lstm_units"])(lstm_layer)
        
        # Process biomarker data
        biomarker_features = layers.Dense(64, activation='relu')(biomarkers_input)
        biomarker_features = layers.Dropout(self.config["dropout_rate"])(biomarker_features)
        
        # Process patient data
        patient_features = layers.Dense(32, activation='relu')(patient_input)
        patient_features = layers.Dropout(self.config["dropout_rate"])(patient_features)
        
        # Process stimulation parameters
        stim_features = layers.Dense(32, activation='relu')(stim_params_input)
        stim_features = layers.Dropout(self.config["dropout_rate"])(stim_features)
        
        # Combine all features
        combined = layers.Concatenate()([electro_features, biomarker_features, patient_features, stim_features])
        
        # Fully connected layers
        x = combined
        for units in self.config["dense_units"]:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.Dropout(self.config["dropout_rate"])(x)
        
        # Output layer - predict recovery trajectory
        if self.use_uncertainty:
            # Output mean and variance for uncertainty quantification
            mean_output = layers.Dense(self.config["prediction_horizon"], name="mean_output")(x)
            log_var_output = layers.Dense(self.config["prediction_horizon"], name="log_var_output")(x)
            
            self.model = keras.Model(
                inputs=[electro_input, biomarkers_input, patient_input, stim_params_input],
                outputs=[mean_output, log_var_output]
            )
            
            # Custom loss function for uncertainty quantification
            def gaussian_nll_loss(y_true, y_pred_mean_logvar):
                y_pred_mean, y_pred_logvar = y_pred_mean_logvar[:, :self.config["prediction_horizon"]], y_pred_mean_logvar[:, self.config["prediction_horizon"]:]
                variance = keras.backend.exp(y_pred_logvar)
                gaussian_nll = keras.backend.sum(0.5 * keras.backend.log(variance) + 0.5 * keras.backend.square(y_true - y_pred_mean) / variance, axis=1)
                return keras.backend.mean(gaussian_nll)
            
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.config["learning_rate"]),
                loss={"mean_output": gaussian_nll_loss, "log_var_output": lambda y_true, y_pred: 0.0},
                metrics={"mean_output": ["mae", "mse"]}
            )
        else:
            # Standard regression output
            output = layers.Dense(self.config["prediction_horizon"], name="recovery_trajectory")(x)
            
            self.model = keras.Model(
                inputs=[electro_input, biomarkers_input, patient_input, stim_params_input],
                outputs=output
            )
            
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.config["learning_rate"]),
                loss="mse",
                metrics=["mae"]
            )
        
        logger.info(f"Built model with {'uncertainty quantification' if self.use_uncertainty else 'standard regression'}")
        self.model.summary(print_fn=logger.info)
    
    def train(
        self,
        X_train: Dict[str, np.ndarray],
        y_train: np.ndarray,
        X_val: Optional[Dict[str, np.ndarray]] = None,
        y_val: Optional[np.ndarray] = None,
        class_weights: Optional[Dict] = None
    ):
        """Train the recovery prediction model.
        
        Args:
            X_train: Dictionary containing training inputs for each input branch
            y_train: Training targets
            X_val: Validation inputs (optional)
            y_val: Validation targets (optional)
            class_weights: Optional weights for imbalanced classes
        
        Returns:
            Training history
        """
        # Prepare callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=self.config["early_stopping_patience"],
                restore_best_weights=True,
                monitor="val_loss" if X_val is not None else "loss"
            ),
            keras.callbacks.ModelCheckpoint(
                filepath="checkpoints/recovery_model_{epoch:02d}_{val_loss:.2f}.h5",
                save_best_only=True,
                monitor="val_loss" if X_val is not None else "loss"
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                monitor="val_loss" if X_val is not None else "loss"
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            x=[
                X_train["electrophysiology"],
                X_train["biomarkers"],
                X_train["patient_data"],
                X_train["stimulation_params"]
            ],
            y=y_train,
            validation_data=[
                [
                    X_val["electrophysiology"],
                    X_val["biomarkers"],
                    X_val["patient_data"],
                    X_val["stimulation_params"]
                ],
                y_val
            ] if X_val is not None and y_val is not None else None,
            epochs=self.config["epochs"],
            batch_size=self.config["batch_size"],
            callbacks=callbacks,
            class_weight=class_weights
        )
        
        return self.history
    
    def predict(
        self,
        X: Dict[str, np.ndarray],
        return_uncertainty: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Make recovery trajectory predictions.
        
        Args:
            X: Dictionary containing model inputs
            return_uncertainty: Whether to return prediction uncertainty
        
        Returns:
            Predicted recovery trajectories and optionally uncertainty estimates
        """
        predictions = self.model.predict([
            X["electrophysiology"],
            X["biomarkers"],
            X["patient_data"],
            X["stimulation_params"]
        ])
        
        if self.use_uncertainty and return_uncertainty:
            mean_preds = predictions[0]
            log_var_preds = predictions[1]
            uncertainty = np.exp(log_var_preds)
            return mean_preds, uncertainty
        else:
            return predictions if not self.use_uncertainty else predictions[0]
    
    def evaluate_feature_importance(self, X: Dict[str, np.ndarray], y: np.ndarray) -> Dict:
        """Evaluate the importance of input features for prediction.
        
        Args:
            X: Dictionary containing model inputs
            y: True target values
        
        Returns:
            Dictionary of feature importance scores
        """
        # Permutation feature importance
        baseline_mae = self.model.evaluate(
            [X["electrophysiology"], X["biomarkers"], X["patient_data"], X["stimulation_params"]],
            y,
            verbose=0
        )[1]  # Get MAE metric
        
        importance_scores = {}
        
        # Check biomarker importance
        for i, biomarker in enumerate(BIOMARKER_WEIGHTS.keys()):
            if i < X["biomarkers"].shape[1]:
                X_permuted = X.copy()
                X_permuted["biomarkers"] = X["biomarkers"].copy()
                X_permuted["biomarkers"][:, i] = np.random.permutation(X_permuted["biomarkers"][:, i])
                
                permuted_mae = self.model.evaluate(
                    [X_permuted["electrophysiology"], X_permuted["biomarkers"], 
                     X_permuted["patient_data"], X_permuted["stimulation_params"]],
                    y,
                    verbose=0
                )[1]  # Get MAE metric
                
                importance = permuted_mae - baseline_mae
                importance_scores[f"biomarker_{biomarker}"] = float(importance)
        
        # Check stimulation parameter importance
        stim_param_names = ["frequency", "amplitude", "pulse_width", "duration", 
                           "waveform", "electrode_config", "timing", "phase"]
        
        for i, param in enumerate(stim_param_names):
            if i < X["stimulation_params"].shape[1]:
                X_permuted = X.copy()
                X_permuted["stimulation_params"] = X["stimulation_params"].copy()
                X_permuted["stimulation_params"][:, i] = np.random.permutation(X_permuted["stimulation_params"][:, i])
                
                permuted_mae = self.model.evaluate(
                    [X_permuted["electrophysiology"], X_permuted["biomarkers"], 
                     X_permuted["patient_data"], X_permuted["stimulation_params"]],
                    y,
                    verbose=0
                )[1]  # Get MAE metric
                
                importance = permuted_mae - baseline_mae
                importance_scores[f"stim_param_{param}"] = float(importance)
        
        self.feature_importance = importance_scores
        return importance_scores
    
    def optimize_stimulation_parameters(
        self,
        patient_data: np.ndarray,
        biomarkers: np.ndarray,
        electrophysiology: np.ndarray,
        target_outcome: float,
        param_bounds: Dict[str, Tuple[float, float]],
        num_iterations: int = 100
    ) -> Dict:
        """Optimize stimulation parameters to achieve target outcome.
        
        Args:
            patient_data: Patient demographic and clinical data
            biomarkers: Current biomarker values
            electrophysiology: Current electrophysiology recordings
            target_outcome: Target recovery outcome value
            param_bounds: Min/max bounds for each stimulation parameter
            num_iterations: Number of optimization iterations
            
        Returns:
            Dictionary of optimized parameters
        """
        # Basic bayesian optimization approach
        best_params = None
        best_score = float('inf')
        
        # Parameter names for the 8 stimulation parameters
        param_names = ["frequency", "amplitude", "pulse_width", "duration", 
                       "waveform", "electrode_config", "timing", "phase"]
        
        # Ensure param_bounds covers all parameters
        for param in param_names:
            if param not in param_bounds:
                param_bounds[param] = (0.0, 1.0)
                logger.warning(f"No bounds provided for {param}, using default (0.0, 1.0)")
        
        # Optimize using random search with bayesian updating
        param_history = []
        score_history = []
        
        for i in range(num_iterations):
            # Generate parameters - with exploration-exploitation tradeoff
            if i < num_iterations // 3:
                # Exploration phase - random sampling
                current_params = np.array([
                    np.random.uniform(param_bounds[param][0], param_bounds[param][1])
                    for param in param_names
                ])
            else:
                # Exploitation phase - sample near best parameters with noise
                exploitation_rate = min(0.8, (i - num_iterations // 3) / (num_iterations * 0.6))
                current_params = np.array([
                    best_params[j] * exploitation_rate + 
                    np.random.uniform(param_bounds[param_names[j]][0], param_bounds[param_names[j]][1]) * (1 - exploitation_rate)
                    for j in range(len(param_names))
                ])
                
                # Ensure parameters are within bounds
                for j, param in enumerate(param_names):
                    current_params[j] = np.clip(
                        current_params[j], 
                        param_bounds[param][0],
                        param_bounds[param][1]
                    )
            
            # Create input for prediction
            X_pred = {
                "electrophysiology": np.expand_dims(electrophysiology, axis=0),
                "biomarkers": np.expand_dims(biomarkers, axis=0),
                "patient_data": np.expand_dims(patient_data, axis=0),
                "stimulation_params": np.expand_dims(current_params, axis=0)
            }
            
            # Predict outcome
            prediction = self.predict(X_pred)
            
            # Use final time step prediction as score
            predicted_outcome = prediction[0, -1]
            score = abs(predicted_outcome - target_outcome)
            
            # Update best parameters if improved
            if score < best_score:
                best_score = score
                best_params = current_params
                logger.info(f"Iteration {i}: New best score: {best_score:.4f}")
            
            param_history.append(current_params)
            score_history.append(score)
        
        # Convert best parameters to dictionary
        optimized_params = {
            param_names[i]: float(best_params[i])
            for i in range(len(param_names))
        }
        
        # Add metadata about optimization
        optimized_params["predicted_outcome"] = float(best_score)
        optimized_params["target_outcome"] = float(target_outcome)
        optimized_params["optimization_iterations"] = num_iterations
        
        return optimized_params
    
    def plot_recovery_trajectory(
        self,
        predictions: np.ndarray,
        uncertainty: Optional[np.ndarray] = None,
        true_values: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ):
        """Plot predicted recovery trajectory with optional uncertainty and ground truth.
        
        Args:
            predictions: Predicted recovery values
            uncertainty: Prediction uncertainty (optional)
            true_values: Ground truth values (optional)
            save_path: Path to save the figure (optional)
        """
        plt.figure(figsize=(10, 6))
        
        time_points = range(self.config["prediction_horizon"])
        
        # Plot predictions
        plt.plot(time_points, predictions, 'b-', label='Predicted Recovery')
        
        # Plot uncertainty if available
        if uncertainty is not None:
            plt.fill_between(
                time_points,
                predictions - 1.96 * np.sqrt(uncertainty),
                predictions + 1.96 * np.sqrt(uncertainty),
                alpha=0.2,
                color='b',
                label='95% Confidence Interval'
            )
        
        # Plot ground truth if available
        if true_values is not None:
            plt.plot(time_points, true_values, 'r--', label='Actual Recovery')
        
        plt.title('Neural Recovery Trajectory Prediction')
        plt.xlabel('Time (weeks)')
        plt.ylabel('Recovery Score')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved recovery trajectory plot to {save_path}")
        
        plt.show()

    def save_model(self, path: str):
        """Save the model weights and configuration.
        
        Args:
            path: Path to save the model
        """
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        
        self.model.save_weights(path)
        
        # Save configuration
        config_path = os.path.join(os.path.dirname(path), "model_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Saved model to {path} and configuration to {config_path}")

# Example usage
if __name__ == "__main__":
    # Create model instance
    recovery_model = RecoveryPredictionModel(use_uncertainty=True)
    
    # Generate some dummy data for testing
    num_samples = 100
    time_steps = recovery_model.config["time_steps"]
    
    # Create dummy input data
    X = {
        "electrophysiology": np.random.randn(num_samples, time_steps, INPUT_FEATURES["electrophysiology"]),
        "biomarkers": np.random.randn(num_samples, INPUT_FEATURES["biomarkers"]),
        "patient_data": np.random.randn(num_samples, INPUT_FEATURES["patient_data"]),
        "stimulation_params": np.random.randn(num_samples, INPUT_FEATURES["stimulation_params"])
    }
    
    # Create dummy target data
    y = np.random.randn(num_samples, recovery_model.config["prediction_horizon"])
    
    # Split data for training and validation
    train_idx = int(0.8 * num_samples)
    X_train = {k: v[:train_idx] for k, v in X.items()}
    y_train = y[:train_idx]
    X_val = {k: v[train_idx:] for k, v in X.items()}
    y_val = y[train_idx:]
    
    # Train model (with minimal epochs for testing)
    recovery_model.config["epochs"] = 2
    recovery_model.train(X_train, y_train, X_val, y_val)
    
    # Make predictions
    if recovery_model.use_uncertainty:
        predictions, uncertainty = recovery_model.predict(X_val, return_uncertainty=True)
        
        # Plot the first sample
        recovery_model.plot_recovery_trajectory(
            predictions[0],
            uncertainty[0],
            y_val[0],
            save_path="recovery_prediction.png"
        )
    else:
        predictions = recovery_model.predict(X_val)
        
        # Plot the first sample
        recovery_model.plot_recovery_trajectory(
            predictions[0],
            None,
            y_val[0],
            save_path="recovery_prediction.png"
        )
    
    # Evaluate feature importance
    importance = recovery_model.evaluate_feature_importance(X_val, y_val)
    print("Feature importance:", importance)
    
    # Optimize stimulation parameters for a sample
    optimized_params = recovery_model.optimize_stimulation_parameters(
        X_val["patient_data"][0],
        X_val["biomarkers"][0],
        X_val["electrophysiology"][0],
        target_outcome=0.8,
        param_bounds={
            "frequency": (0.1, 100.0),
            "amplitude": (0.1, 5.0),
            "pulse_width": (0.1, 1.0),
            "duration": (5.0, 60.0),
            "waveform": (0.0, 1.0),
            "electrode_config": (0.0, 1.0),
            "timing": (0.0, 1.0),
            "phase": (0.0, 1.0)
        },
        num_iterations=50
    )
    
    print("Optimized parameters:", optimized_params)