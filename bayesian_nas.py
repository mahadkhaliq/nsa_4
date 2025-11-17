"""
Bayesian Optimization for Neural Architecture Search

This module implements Bayesian optimization for searching optimal
approximate multiplier configurations and architectures for AxDNNs.

Uses Gaussian Process to model the relationship between architecture
parameters and objective function (accuracy, energy, or STL robustness).
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
import random


class BayesianNAS:
    """Bayesian Optimization for NAS

    Uses Gaussian Process to efficiently search architecture space by:
    1. Sampling a few random architectures initially
    2. Fitting GP model to predict performance
    3. Using acquisition function to select next promising architecture
    4. Iteratively improving until convergence
    """

    def __init__(self, search_space, objective='accuracy',
                 n_initial=5, n_iterations=15, xi=0.01):
        """Initialize Bayesian NAS

        Args:
            search_space: Dict defining search space
            objective: 'accuracy', 'energy', or 'stl_robustness'
            n_initial: Number of random samples to start
            n_iterations: Number of Bayesian optimization iterations
            xi: Exploration-exploitation tradeoff (higher = more exploration)
        """
        self.search_space = search_space
        self.objective = objective
        self.n_initial = n_initial
        n_iterations = n_iterations
        self.xi = xi

        # GP model for Bayesian optimization
        kernel = Matern(nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5
        )

        # History of evaluations
        self.X_observed = []  # Architecture encodings
        self.y_observed = []  # Objective values

    def encode_architecture(self, arch):
        """Encode architecture dict as numerical vector for GP

        Args:
            arch: Architecture dictionary

        Returns:
            encoding: Numpy array of numerical features
        """
        encoding = []

        # Encode architecture parameters
        if 'num_stages' in arch:
            # ResNet architecture
            encoding.append(arch['num_stages'])
            encoding.append(arch['blocks_per_stage'])
            encoding.append(arch['base_filters'])

            # Encode multipliers (use index in multiplier list)
            mul_options = self.search_space['mul_map_files']
            for mul in arch['mul_map_files']:
                try:
                    encoding.append(mul_options.index(mul))
                except ValueError:
                    encoding.append(0)  # Default if not found
        else:
            # CNN architecture
            encoding.append(arch['num_conv_layers'])
            encoding.extend(arch['filters'][:4])  # Pad if needed
            encoding.extend(arch['kernels'][:4])
            encoding.append(arch['dense_units'])

            # Encode multipliers
            mul_options = self.search_space['mul_map_files']
            for mul in arch['mul_map_files'][:4]:
                try:
                    encoding.append(mul_options.index(mul))
                except ValueError:
                    encoding.append(0)

        return np.array(encoding, dtype=float)

    def sample_random_architecture(self):
        """Sample random architecture from search space"""
        if 'num_stages' in self.search_space:
            # ResNet sampling
            from nas_search import sample_resnet_multipliers
            return sample_resnet_multipliers(self.search_space)
        else:
            # CNN sampling
            from nas_search import sample_architecture
            return sample_architecture(self.search_space)

    def acquisition_function(self, X, X_sample, y_sample):
        """Expected Improvement acquisition function

        Args:
            X: Candidate architecture encodings to evaluate
            X_sample: Observed architecture encodings
            y_sample: Observed objective values

        Returns:
            ei_values: Expected improvement for each candidate
        """
        # Fit GP if we have enough samples
        if len(X_sample) < 2:
            return np.random.rand(len(X))

        self.gp.fit(X_sample, y_sample)

        # Predict mean and std for candidates
        mu, sigma = self.gp.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1)

        # Current best observed value
        mu_sample_opt = np.max(y_sample)

        # Expected Improvement calculation
        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - self.xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei.flatten()

    def suggest_next_architecture(self):
        """Suggest next architecture to evaluate using Bayesian optimization

        Returns:
            arch: Promising architecture to evaluate next
        """
        # Generate random candidate architectures
        n_candidates = 100
        candidates = [self.sample_random_architecture() for _ in range(n_candidates)]
        X_candidates = np.array([self.encode_architecture(arch) for arch in candidates])

        # If we have observations, use acquisition function
        if len(self.X_observed) >= self.n_initial:
            X_sample = np.array(self.X_observed)
            y_sample = np.array(self.y_observed)

            # Calculate expected improvement
            ei_values = self.acquisition_function(X_candidates, X_sample, y_sample)

            # Select candidate with highest EI
            best_idx = np.argmax(ei_values)
            return candidates[best_idx]
        else:
            # Random sampling for initial observations
            return random.choice(candidates)

    def update_observations(self, arch, result):
        """Update observations with new evaluation result

        Args:
            arch: Architecture evaluated
            result: Result dict with 'approx_accuracy', 'energy', 'stl_robustness'
        """
        encoding = self.encode_architecture(arch)

        # Extract objective value
        if self.objective == 'accuracy':
            value = result['approx_accuracy'] if result['approx_accuracy'] else 0.0
        elif self.objective == 'energy':
            value = -result['energy'] if result['energy'] else 0.0  # Negative for minimization
        elif self.objective == 'stl_robustness':
            value = result['stl_robustness'] if result['stl_robustness'] else -1000.0
        else:
            value = result.get(self.objective, 0.0)

        self.X_observed.append(encoding)
        self.y_observed.append(value)

    def get_best_architecture(self):
        """Get best architecture found so far

        Returns:
            best_idx: Index of best architecture in observations
        """
        if not self.y_observed:
            return None
        return np.argmax(self.y_observed)


def bayesian_search(search_space, num_trials=20, objective='accuracy'):
    """Run Bayesian optimization for architecture search

    Args:
        search_space: Dict defining search space
        num_trials: Total number of architectures to evaluate
        objective: 'accuracy', 'energy', or 'stl_robustness'

    Returns:
        List of architectures suggested by Bayesian optimization
    """
    # Initialize Bayesian NAS
    n_initial = min(5, num_trials // 4)  # 25% random initialization
    bayes_nas = BayesianNAS(
        search_space=search_space,
        objective=objective,
        n_initial=n_initial,
        n_iterations=num_trials - n_initial
    )

    # Generate architectures to evaluate
    architectures = []
    for i in range(num_trials):
        arch = bayes_nas.suggest_next_architecture()
        architectures.append(arch)

    return architectures, bayes_nas
