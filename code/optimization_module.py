"""
Optimization Module for Neural Stimulation Parameter Optimization

This module implements various optimization algorithms for finding optimal 
stimulation parameters based on the neural response.

Features:
- Bayesian optimization
- Grid search and random search
- Genetic algorithm
- Parameter space exploration and visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from scipy.stats import norm
import pickle
import os
import time
import json
from datetime import datetime
import itertools
import random


class ParameterOptimizer:
    """Base class for parameter optimization"""
    
    def __init__(self, param_space, objective_func, maximize=True):
        """
        Initialize the parameter optimizer
        
        Parameters:
        -----------
        param_space : dict
            Dictionary of parameter ranges {param_name: (min, max, step)}
        objective_func : callable
            Function to evaluate parameter sets (params -> score)
        maximize : bool
            Whether to maximize (True) or minimize (False) the objective
        """
        self.param_space = param_space
        self.objective_func = objective_func
        self.maximize = maximize
        
        # History of evaluations
        self.eval_history = []
        
        # Best parameters so far
        self.best_params = None
        self.best_score = float('-inf') if maximize else float('inf')
    
    def optimize(self, n_iterations=10):
        """
        Run the optimization
        
        Parameters:
        -----------
        n_iterations : int
            Number of iterations to run
            
        Returns:
        --------
        best_params : dict
            Best parameter set found
        best_score : float
            Best score achieved
        """
        raise NotImplementedError("Subclasses must implement optimize method")
    
    def log_evaluation(self, params, score):
        """
        Log a parameter evaluation
        
        Parameters:
        -----------
        params : dict
            Parameter set
        score : float
            Score achieved
        """
        # Check if this is the best score so far
        is_better = False
        if self.maximize:
            is_better = score > self.best_score
        else:
            is_better = score < self.best_score
        
        if is_better:
            self.best_score = score
            self.best_params = params.copy()
        
        # Log the evaluation
        self.eval_history.append({
            'params': params.copy(),
            'score': score,
            'timestamp': time.time(),
            'is_best': is_better
        })
    
    def get_best(self):
        """
        Get the best parameters found
        
        Returns:
        --------
        best_params : dict
            Best parameter set found
        best_score : float
            Best score achieved
        """
        return self.best_params, self.best_score
    
    def get_history(self):
        """
        Get the evaluation history
        
        Returns:
        --------
        history : list
            List of evaluation records
        """
        return self.eval_history
    
    def save_results(self, filepath):
        """
        Save optimization results to a file
        
        Parameters:
        -----------
        filepath : str
            Path to save the results
        """
        results = {
            'param_space': self.param_space,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'eval_history': self.eval_history,
            'maximize': self.maximize,
            'optimizer_type': self.__class__.__name__,
            'timestamp': datetime.now().isoformat()
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save as pickle
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    
    def load_results(self, filepath):
        """
        Load optimization results from a file
        
        Parameters:
        -----------
        filepath : str
            Path to load the results from
            
        Returns:
        --------
        success : bool
            Whether loading was successful
        """
        if not os.path.exists(filepath):
            return False
        
        try:
            with open(filepath, 'rb') as f:
                results = pickle.load(f)
            
            # Load results
            self.param_space = results['param_space']
            self.best_params = results['best_params']
            self.best_score = results['best_score']
            self.eval_history = results['eval_history']
            self.maximize = results['maximize']
            
            return True
        except Exception as e:
            print(f"Error loading results: {e}")
            return False
    
    def plot_convergence(self):
        """
        Plot optimization convergence
        
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure containing the plot
        """
        if not self.eval_history:
            print("No evaluation history to plot")
            return None
        
        # Extract data
        iterations = range(1, len(self.eval_history) + 1)
        scores = [record['score'] for record in self.eval_history]
        
        # If minimizing, negate scores for visualization
        if not self.maximize:
            scores = [-score for score in scores]
        
        # Calculate running best
        running_best = []
        current_best = float('-inf')
        for score in scores:
            current_best = max(current_best, score)
            running_best.append(current_best)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot scores
        ax.scatter(iterations, scores, alpha=0.6, label='Evaluations')
        
        # Plot running best
        ax.plot(iterations, running_best, 'r-', label='Best so far')
        
        # Add labels and title
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective Value (higher is better)')
        ax.set_title('Optimization Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_parameter_importance(self):
        """
        Plot parameter importance based on correlation with the objective
        
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure containing the plot
        """
        if not self.eval_history or len(self.eval_history) < 5:
            print("Not enough evaluation history to analyze parameter importance")
            return None
        
        # Create a DataFrame from evaluation history
        data = []
        for record in self.eval_history:
            row = {**record['params'], 'score': record['score']}
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Calculate correlation with the objective
        param_cols = list(self.param_space.keys())
        correlations = []
        
        for param in param_cols:
            corr = df[[param, 'score']].corr().iloc[0, 1]
            correlations.append((param, corr))
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot correlations
        params = [x[0] for x in correlations]
        corrs = [x[1] for x in correlations]
        
        # Create colormap based on sign
        colors = ['green' if c > 0 else 'red' for c in corrs]
        
        ax.barh(params, corrs, color=colors)
        
        # Add labels and title
        ax.set_xlabel('Correlation with Objective')
        ax.set_title('Parameter Importance')
        ax.grid(True, alpha=0.3)
        
        # Add a line at zero
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        return fig


class GridSearchOptimizer(ParameterOptimizer):
    """Grid search optimizer for parameter optimization"""
    
    def __init__(self, param_space, objective_func, maximize=True, exhaustive=False):
        """
        Initialize the grid search optimizer
        
        Parameters:
        -----------
        param_space : dict
            Dictionary of parameter ranges {param_name: (min, max, step)}
        objective_func : callable
            Function to evaluate parameter sets (params -> score)
        maximize : bool
            Whether to maximize (True) or minimize (False) the objective
        exhaustive : bool
            Whether to search the entire grid (True) or use a coarser grid (False)
        """
        super().__init__(param_space, objective_func, maximize)
        self.exhaustive = exhaustive
    
    def optimize(self, n_iterations=None):
        """
        Run grid search optimization
        
        Parameters:
        -----------
        n_iterations : int
            Number of iterations to run (ignored for exhaustive grid search)
            
        Returns:
        --------
        best_params : dict
            Best parameter set found
        best_score : float
            Best score achieved
        """
        # Define the grid
        param_values = {}
        for param, (min_val, max_val, step) in self.param_space.items():
            if self.exhaustive:
                # Use the full grid with the specified step size
                values = np.arange(min_val, max_val + step, step)
            else:
                # Use a coarser grid
                n_values = min(5, int((max_val - min_val) / step) + 1)
                values = np.linspace(min_val, max_val, n_values)
            
            param_values[param] = values
        
        # Create all combinations
        param_names = list(param_values.keys())
        param_grids = list(itertools.product(*[param_values[param] for param in param_names]))
        
        # If n_iterations is specified and less than the total grid size,
        # randomly sample from the grid
        if n_iterations is not None and n_iterations < len(param_grids):
            param_grids = random.sample(param_grids, n_iterations)
        
        print(f"Grid search with {len(param_grids)} parameter combinations")
        
        # Evaluate all grid points
        for i, grid_point in enumerate(param_grids):
            params = {param: value for param, value in zip(param_names, grid_point)}
            
            # Evaluate the parameter set
            score = self.objective_func(params)
            
            # Log the evaluation
            self.log_evaluation(params, score)
            
            # Print progress
            if (i + 1) % 10 == 0 or i == len(param_grids) - 1:
                print(f"Evaluated {i + 1}/{len(param_grids)} combinations")
        
        return self.best_params, self.best_score


class RandomSearchOptimizer(ParameterOptimizer):
    """Random search optimizer for parameter optimization"""
    
    def optimize(self, n_iterations=100):
        """
        Run random search optimization
        
        Parameters:
        -----------
        n_iterations : int
            Number of iterations to run
            
        Returns:
        --------
        best_params : dict
            Best parameter set found
        best_score : float
            Best score achieved
        """
        print(f"Random search with {n_iterations} iterations")
        
        for i in range(n_iterations):
            # Generate random parameter set
            params = {}
            for param, (min_val, max_val, step) in self.param_space.items():
                # Generate a random value within the range
                if isinstance(min_val, int) and isinstance(max_val, int) and isinstance(step, int):
                    # For integer parameters
                    num_steps = int((max_val - min_val) / step) + 1
                    step_idx = random.randint(0, num_steps - 1)
                    value = min_val + step_idx * step
                else:
                    # For float parameters
                    value = random.uniform(min_val, max_val)
                    # Quantize to step size
                    steps = round((value - min_val) / step)
                    value = min_val + steps * step
                
                params[param] = value
            
            # Evaluate the parameter set
            score = self.objective_func(params)
            
            # Log the evaluation
            self.log_evaluation(params, score)
            
            # Print progress
            if (i + 1) % 10 == 0 or i == n_iterations - 1:
                print(f"Completed {i + 1}/{n_iterations} iterations")
        
        return self.best_params, self.best_score


class BayesianOptimizer(ParameterOptimizer):
    """Bayesian optimizer using Gaussian Processes for parameter optimization"""
    
    def __init__(self, param_space, objective_func, maximize=True, 
                 exploration_weight=0.1, kernel=None):
        """
        Initialize the Bayesian optimizer
        
        Parameters:
        -----------
        param_space : dict
            Dictionary of parameter ranges {param_name: (min, max, step)}
        objective_func : callable
            Function to evaluate parameter sets (params -> score)
        maximize : bool
            Whether to maximize (True) or minimize (False) the objective
        exploration_weight : float
            Weight for exploration vs exploitation (higher values encourage exploration)
        kernel : sklearn.gaussian_process.kernels.Kernel
            Kernel for the Gaussian Process (default: RBF kernel)
        """
        super().__init__(param_space, objective_func, maximize)
        self.exploration_weight = exploration_weight
        
        # Set default kernel if not provided
        if kernel is None:
            param_names = list(param_space.keys())
            kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[1.0] * len(param_names),
                                                 length_scale_bounds=[(1e-2, 1e2)] * len(param_names),
                                                 nu=2.5)
        
        self.kernel = kernel
        
        # Initialize Gaussian Process
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=1e-6,  # Noise level
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=42
        )
        
        # Initialize parameter scaling
        self.scaler = StandardScaler()
        self.X_scaled = None
        self.y = None
    
    def _params_to_array(self, params):
        """Convert parameter dictionary to array"""
        return np.array([params[param] for param in self.param_space.keys()])
    
    def _array_to_params(self, x):
        """Convert array to parameter dictionary"""
        return {param: x[i] for i, param in enumerate(self.param_space.keys())}
    
    def _expected_improvement(self, X, xi=0.01):
        """
        Calculate expected improvement at X
        
        Parameters:
        -----------
        X : ndarray
            Points to calculate EI at
        xi : float
            Exploration-exploitation trade-off parameter
            
        Returns:
        --------
        ei : ndarray
            Expected improvement at X
        """
        # Make prediction
        mu, sigma = self.gp.predict(X, return_std=True)
        
        # Calculate improvement
        if self.maximize:
            # For maximization
            imp = mu - self.best_score - xi
        else:
            # For minimization
            imp = self.best_score - mu - xi
        
        # Calculate EI
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        
        # If sigma is 0, EI is 0
        ei[sigma == 0.0] = 0.0
        
        return ei
    
    def _sample_next_point(self):
        """
        Sample the next point to evaluate
        
        Returns:
        --------
        next_params : dict
            Next parameter set to evaluate
        """
        param_names = list(self.param_space.keys())
        n_params = len(param_names)
        
        # Define bounds
        bounds = [(self.param_space[param][0], self.param_space[param][1]) 
                 for param in param_names]
        
        # Try different starting points
        best_x = None
        best_ei = -1
        
        # Start from random points
        for _ in range(10):
            # Random starting point
            x0 = np.array([random.uniform(bounds[i][0], bounds[i][1]) 
                          for i in range(n_params)])
            
            # Convert to right format for scipy.optimize
            scipy_bounds = [(bounds[i][0], bounds[i][1]) for i in range(n_params)]
            
            # Define objective function (negative because scipy minimizes)
            def objective(x):
                # Reshape to 2D array for GP
                x_2d = x.reshape(1, -1)
                
                # Scale the input
                if self.X_scaled is not None:
                    x_scaled = self.scaler.transform(x_2d)
                else:
                    x_scaled = x_2d
                
                # Return negative EI
                return -self._expected_improvement(x_scaled)[0]
            
            # Run optimization
            result = minimize(objective, x0, bounds=scipy_bounds, method='L-BFGS-B')
            
            # Check if this is better than previous result
            if result.success and -result.fun > best_ei:
                best_ei = -result.fun
                best_x = result.x
        
        # If no good point found, use random sample
        if best_x is None:
            best_x = np.array([random.uniform(bounds[i][0], bounds[i][1]) 
                              for i in range(n_params)])
        
        # Convert to parameters
        next_params = {param: best_x[i] for i, param in enumerate(param_names)}
        
        # Quantize parameters
        for param, (min_val, max_val, step) in self.param_space.items():
            steps = round((next_params[param] - min_val) / step)
            next_params[param] = min_val + steps * step
            # Ensure within bounds
            next_params[param] = max(min_val, min(next_params[param], max_val))
        
        return next_params
    
    def optimize(self, n_iterations=50, n_random_init=10):
        """
        Run Bayesian optimization
        
        Parameters:
        -----------
        n_iterations : int
            Number of iterations to run
        n_random_init : int
            Number of random evaluations before using the model
            
        Returns:
        --------
        best_params : dict
            Best parameter set found
        best_score : float
            Best score achieved
        """
        print(f"Bayesian optimization with {n_iterations} iterations")
        
        # Initial random evaluations
        print(f"Performing {n_random_init} random evaluations...")
        for i in range(n_random_init):
            # Generate random parameter set
            params = {}
            for param, (min_val, max_val, step) in self.param_space.items():
                # Generate a random value within the range
                value = random.uniform(min_val, max_val)
                # Quantize to step size
                steps = round((value - min_val) / step)
                value = min_val + steps * step
                params[param] = value
            
            # Evaluate the parameter set
            score = self.objective_func(params)
            
            # Log the evaluation
            self.log_evaluation(params, score)
        
        # Prepare data for GP
        X = np.array([self._params_to_array(record['params']) for record in self.eval_history])
        y = np.array([record['score'] for record in self.eval_history])
        
        # If minimizing, negate y for the GP
        if not self.maximize:
            y = -y
        
        # Scale X
        self.X_scaled = self.scaler.fit_transform(X)
        self.y = y
        
        # Fit GP
        self.gp.fit(self.X_scaled, self.y)
        
        # Main optimization loop
        for i in range(n_iterations - n_random_init):
            # Sample next point
            next_params = self._sample_next_point()
            
            # Evaluate the parameter set
            score = self.objective_func(next_params)
            
            # Log the evaluation
            self.log_evaluation(next_params, score)
            
            # Update the GP model
            X = np.array([self._params_to_array(record['params']) for record in self.eval_history])
            y = np.array([record['score'] for record in self.eval_history])
            
            # If minimizing, negate y for the GP
            if not self.maximize:
                y = -y
            
            # Scale X
            self.X_scaled = self.scaler.fit_transform(X)
            self.y = y
            
            # Fit GP
            self.gp.fit(self.X_scaled, self.y)
            
            # Print progress
            if (i + 1) % 5 == 0 or i == n_iterations - n_random_init - 1:
                print(f"Completed {i + 1 + n_random_init}/{n_iterations} iterations")
        
        return self.best_params, self.best_score


class GeneticOptimizer(ParameterOptimizer):
    """Genetic algorithm optimizer for parameter optimization"""
    
    def __init__(self, param_space, objective_func, maximize=True, 
                 population_size=20, mutation_rate=0.1, crossover_rate=0.7):
        """
        Initialize the genetic optimizer
        
        Parameters:
        -----------
        param_space : dict
            Dictionary of parameter ranges {param_name: (min, max, step)}
        objective_func : callable
            Function to evaluate parameter sets (params -> score)
        maximize : bool
            Whether to maximize (True) or minimize (False) the objective
        population_size : int
            Size of the population
        mutation_rate : float
            Probability of mutation (0-1)
        crossover_rate : float
            Probability of crossover (0-1)
        """
        super().__init__(param_space, objective_func, maximize)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
    
    def _initialize_population(self):
        """
        Initialize a random population
        
        Returns:
        --------
        population : list
            List of parameter sets
        """
        population = []
        
        for _ in range(self.population_size):
            # Generate random parameter set
            params = {}
            for param, (min_val, max_val, step) in self.param_space.items():
                # Generate a random value within the range
                if isinstance(min_val, int) and isinstance(max_val, int) and isinstance(step, int):
                    # For integer parameters
                    num_steps = int((max_val - min_val) / step) + 1
                    step_idx = random.randint(0, num_steps - 1)
                    value = min_val + step_idx * step
                else:
                    # For float parameters
                    value = random.uniform(min_val, max_val)
                    # Quantize to step size
                    steps = round((value - min_val) / step)
                    value = min_val + steps * step
                
                params[param] = value
            
            population.append(params)
        
        return population
    
    def _evaluate_population(self, population):
        """
        Evaluate all individuals in the population
        
        Parameters:
        -----------
        population : list
            List of parameter sets
            
        Returns:
        --------
        fitness : list
            List of fitness scores
        """
        fitness = []
        
        for params in population:
            score = self.objective_func(params)
            self.log_evaluation(params, score)
            fitness.append(score)
        
        return fitness
    
    def _select_parents(self, population, fitness):
        """
        Select parents for reproduction using tournament selection
        
        Parameters:
        -----------
        population : list
            List of parameter sets
        fitness : list
            List of fitness scores
            
        Returns:
        --------
        parents : list
            List of selected parameter sets
        """
        parents = []
        
        # Number of parents to select (should be even for crossover)
        n_parents = self.population_size
        if n_parents % 2 != 0:
            n_parents -= 1
        
        # Tournament selection
        for _ in range(n_parents):
            # Select two random individuals
            idx1, idx2 = random.sample(range(len(population)), 2)
            
            # Compare fitness
            if self.maximize:
                if fitness[idx1] > fitness[idx2]:
                    parents.append(population[idx1])
                else:
                    parents.append(population[idx2])
            else:
                if fitness[idx1] < fitness[idx2]:
                    parents.append(population[idx1])
                else:
                    parents.append(population[idx2])
        
        return parents
    
    def _crossover(self, parent1, parent2):
        """
        Perform crossover between two parents
        
        Parameters:
        -----------
        parent1, parent2 : dict
            Parent parameter sets
            
        Returns:
        --------
        child1, child2 : dict
            Child parameter sets
        """
        if random.random() > self.crossover_rate:
            # No crossover, return copies of parents
            return parent1.copy(), parent2.copy()
        
        # Perform crossover
        child1 = {}
        child2 = {}
        
        # For each parameter, randomly choose which parent to inherit from
        for param in self.param_space.keys():
            if random.random() < 0.5:
                child1[param] = parent1[param]
                child2[param] = parent2[param]
            else:
                child1[param] = parent2[param]
                child2[param] = parent1[param]
        
        return child1, child2
    
    def _mutate(self, individual):
        """
        Perform mutation on an individual
        
        Parameters:
        -----------
        individual : dict
            Parameter set
            
        Returns:
        --------
        mutated : dict
            Mutated parameter set
        """
        mutated = individual.copy()
        
        for param, (min_val, max_val, step) in self.param_space.items():
            # Apply mutation with probability
            if random.random() < self.mutation_rate:
                # Random mutation within a smaller range
                range_size = max_val - min_val
                mutation_range = range_size * 0.2  # Mutate within 20% of range
                
                # Mutate by adding/subtracting a random value
                delta = random.uniform(-mutation_range, mutation_range)
                new_value = mutated[param] + delta
                
                # Ensure within bounds
                new_value = max(min_val, min(new_value, max_val))
                
                # Quantize to step size
                steps = round((new_value - min_val) / step)
                mutated[param] = min_val + steps * step
        
        return mutated
    
    def optimize(self, n_iterations=50):
        """
        Run genetic optimization
        
        Parameters:
        -----------
        n_iterations : int
            Number of generations to run
            
        Returns:
        --------
        best_params : dict
            Best parameter set found
        best_score : float
            Best score achieved
        """
        print(f"Genetic optimization with {n_iterations} generations")
        
        # Initialize population
        population = self._initialize_population()
        
        # Evaluate initial population
        fitness = self._evaluate_population(population)
        
        # Main optimization loop
        for generation in range(n_iterations):
            # Select parents
            parents = self._select_parents(population, fitness)
            
            # Create next generation
            next_population = []
            
            # Elitism: keep the best individual
            if self.maximize:
                best_idx = np.argmax(fitness)
            else:
                best_idx = np.argmin(fitness)
            
            next_population.append(population[best_idx])
            
            # Crossover and mutation
            for i in range(0, len(parents) - 1, 2):
                # Crossover
                child1, child2 = self._crossover(parents[i], parents[i + 1])
                
                # Mutation
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                # Add to next generation
                next_population.append(child1)
                next_population.append(child2)
            
            # Ensure population size is maintained
            while len(next_population) < self.population_size:
                # Add random individuals
                next_population.append(self._mutate(random.choice(population)))
            
            # Update population
            population = next_population[:self.population_size]
            
            # Evaluate new population
            fitness = self._evaluate_population(population)
            
            # Print progress
            if (generation + 1) % 5 == 0 or generation == n_iterations - 1:
                if self.maximize:
                    best_fitness = max(fitness)
                else:
                    best_fitness = min(fitness)
                print(f"Generation {generation + 1}/{n_iterations}, Best Fitness: {best_fitness:.4f}")
        
        return self.best_params, self.best_score


# Function to create an optimizer
def create_optimizer(method, param_space, objective_func, maximize=True, **kwargs):
    """
    Create an optimizer based on the specified method
    
    Parameters:
    -----------
    method : str
        Optimization method ('grid', 'random', 'bayesian', 'genetic')
    param_space : dict
        Dictionary of parameter ranges {param_name: (min, max, step)}
    objective_func : callable
        Function to evaluate parameter sets (params -> score)
    maximize : bool
        Whether to maximize (True) or minimize (False) the objective
    **kwargs : dict
        Additional arguments for the specific optimizer
        
    Returns:
    --------
    optimizer : ParameterOptimizer
        The created optimizer
    """
    if method == 'grid':
        return GridSearchOptimizer(param_space, objective_func, maximize, **kwargs)
    elif method == 'random':
        return RandomSearchOptimizer(param_space, objective_func, maximize, **kwargs)
    elif method == 'bayesian':
        return BayesianOptimizer(param_space, objective_func, maximize, **kwargs)
    elif method == 'genetic':
        return GeneticOptimizer(param_space, objective_func, maximize, **kwargs)
    else:
        raise ValueError(f"Unknown optimization method: {method}")


# Example usage
if __name__ == "__main__":
    # Define parameter space
    param_space = {
        'frequency': (10, 200, 5),      # Hz (min, max, step)
        'amplitude': (0.5, 5.0, 0.1),   # mA
        'pulse_width': (50, 500, 10),   # µs
        'duty_cycle': (10, 100, 5)      # %
    }
    
    # Define objective function (example)
    def objective_func(params):
        # This is a dummy objective function simulating the stimulation effect
        # In reality, this would interact with the actual system
        
        # Simulating a function with a peak around certain values
        freq_term = -((params['frequency'] - 100) / 100) ** 2
        amp_term = -((params['amplitude'] - 2.5) / 2.5) ** 2
        pw_term = -((params['pulse_width'] - 300) / 300) ** 2
        duty_term = -((params['duty_cycle'] - 50) / 50) ** 2
        
        # Combined objective with some noise
        objective = (0.4 * freq_term + 0.3 * amp_term + 
                     0.2 * pw_term + 0.1 * duty_term)
        
        # Scale to 0-100 and add noise
        score = 100 * (objective + 1) / 2 + np.random.normal(0, 2)
        
        return max(0, min(100, score))
    
    # Create and run a Bayesian optimizer
    optimizer = BayesianOptimizer(param_space, objective_func, maximize=True)
    best_params, best_score = optimizer.optimize(n_iterations=30)
    
    print("\nOptimization Results:")
    print(f"Best Score: {best_score:.2f}")
    print("Best Parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Plot the results
    optimizer.plot_convergence()
    plt.show()
