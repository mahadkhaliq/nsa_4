
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
import random


class BayesianNAS:

    def __init__(self, search_space, objective='accuracy',
                 n_initial=5, n_iterations=15, xi=0.01):
        self.search_space = search_space
        self.objective = objective
        self.n_initial = n_initial
        n_iterations = n_iterations
        self.xi = xi

        kernel = Matern(nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5
        )

        self.X_observed = []
        self.y_observed = []

    def encode_architecture(self, arch):
        encoding = []

        if 'num_stages' in arch:
            encoding.append(arch['num_stages'])

            blocks_per_stage_raw = arch['blocks_per_stage']
            if isinstance(blocks_per_stage_raw, int):
                blocks_list = [blocks_per_stage_raw] * arch['num_stages']
            else:
                blocks_list = blocks_per_stage_raw

            for i in range(4):
                if i < len(blocks_list):
                    encoding.append(blocks_list[i])
                else:
                    encoding.append(0)

            encoding.append(arch['filters_per_stage'][0])

            mul_options = self.search_space['mul_map_files']
            for mul in arch['mul_map_files']:
                try:
                    encoding.append(mul_options.index(mul))
                except ValueError:
                    encoding.append(0)
            for i in range(4 - len(arch['mul_map_files'])):
                encoding.append(0)
        else:
            encoding.append(arch['num_conv_layers'])
            encoding.extend(arch['filters'][:4])
            encoding.extend(arch['kernels'][:4])
            encoding.append(arch['dense_units'])

            mul_options = self.search_space['mul_map_files']
            for mul in arch['mul_map_files'][:4]:
                try:
                    encoding.append(mul_options.index(mul))
                except ValueError:
                    encoding.append(0)

        return np.array(encoding, dtype=float)

    def sample_random_architecture(self):
        if 'num_stages' in self.search_space:
            from nas_search import sample_resnet_multipliers
            return sample_resnet_multipliers(self.search_space)
        else:
            from nas_search import sample_architecture
            return sample_architecture(self.search_space)

    def acquisition_function(self, X, X_sample, y_sample):
        if len(X_sample) < 2:
            return np.random.rand(len(X))

        self.gp.fit(X_sample, y_sample)

        mu, sigma = self.gp.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1)

        mu_sample_opt = np.max(y_sample)

        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - self.xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei.flatten()

    def suggest_next_architecture(self):
        n_candidates = 100
        candidates = [self.sample_random_architecture() for _ in range(n_candidates)]
        X_candidates = np.array([self.encode_architecture(arch) for arch in candidates])

        if len(self.X_observed) >= self.n_initial:
            X_sample = np.array(self.X_observed)
            y_sample = np.array(self.y_observed)

            ei_values = self.acquisition_function(X_candidates, X_sample, y_sample)

            best_idx = np.argmax(ei_values)
            return candidates[best_idx]
        else:
            return random.choice(candidates)

    def update_observations(self, arch, result):
        encoding = self.encode_architecture(arch)

        if self.objective == 'accuracy':
            value = result['approx_accuracy'] if result['approx_accuracy'] else 0.0
        elif self.objective == 'energy':
            value = -result['energy'] if result['energy'] else 0.0
        elif self.objective == 'stl_robustness':
            value = result['stl_robustness'] if result['stl_robustness'] else -1000.0
        else:
            value = result.get(self.objective, 0.0)

        self.X_observed.append(encoding)
        self.y_observed.append(value)

    def get_best_architecture(self):
        if not self.y_observed:
            return None
        return np.argmax(self.y_observed)


def bayesian_search(search_space, num_trials=20, objective='accuracy'):
    n_initial = min(5, num_trials // 4)
    bayes_nas = BayesianNAS(
        search_space=search_space,
        objective=objective,
        n_initial=n_initial,
        n_iterations=num_trials - n_initial
    )

    architectures = []
    for i in range(num_trials):
        arch = bayes_nas.suggest_next_architecture()
        architectures.append(arch)

    return architectures, bayes_nas
