from .surfaces import ConstraintSurface

from typing import Optional, Callable

import torch
import numpy as np
from scipy import stats

class ManifoldMCMCSampler:

    def __init__(self, n_dim: int, scale: float, constraint_equation: Callable, metric: Optional[Callable]=None, density_function: Optional[Callable]=None) -> None:
        self.n_dim = n_dim
        self.surface = ConstraintSurface(constraint_equation, metric)
        self.scale = scale

        if metric:
            self.metric = metric
        else:
            self.metric =  _euclidean_metric

        if density_function:
            self.density_function = density_function
        else:
            self.density_function = lambda *args: 1.0

        
    def step(self, current_point):
        # Generate orthogonal basis for tangent plane and normal 
        tangent_space = self.surface.generate_tangent_space(current_point)
        normal_space = self.surface.generate_normal_space(current_point)

        # Sample tangent plane
        metric_tensor = self.metric(current_point)
        covariance = torch.inverse(metric_tensor) * self.scale
        sample = np.random.multivariate_normal(mean=np.zeros(len(tangent_space)), cov=covariance)
        v = sample * tangent_space
        p_v = stats.multivariate_normal.pdf(sample, mean=np.zeros(len(tangent_space)), cov=covariance)

        # Solve for projected point 
        new_point = _project(current_point + v, self.surface, normal_space)

        # If projection fails, reject
        if not new_point:
            return current_point

        # Generate normal and tangent space for new point
        new_tangent_space = self.surface.generate_tangent_space(new_point)
        new_normal_space = self.surface.generate_normal_space(new_point)

        # Find v' and p(v') for reverse projection step
        v_prime, w_prime, p_v_prime = self._reverse_projection(current_point, new_point, new_normal_space, new_tangent_space)

        # Check reverse projection works
        if not v_prime:
            return current_point

        # Metropolis-Hastings rejection step
        u = np.random.random()
        acceptance_prob = (self.density_function(new_point) * p_v_prime) / (self.density_function(current_point) * p_v)
        if u > acceptance_prob:
            return current_point

        return new_point

    def sample(self, n_samples, initial_point=None, log_interval:Optional[int]=None):
        if not initial_point:
            initial_point = self._get_initial_point()

        if not log_interval:
            log_interval = n_samples

        current_point = initial_point
        samples = [current_point]
        for i in range(n_samples):
            if i % log_interval == 0:
                print(f"step {i}")
            current_point = self.step(current_point)
            samples.append(current_point)

    def _get_intial_point(self):
        pass

    def _reverse_projection(self, current_point, new_point, new_normal_space, new_tangent_space):
        pass


def _project(point, surface, normal_space):
    pass

def _euclidean_metric(x):
    return torch.eye(x.shape[0])