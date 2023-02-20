from geometric_sampling.manifold_sampling.surfaces import ConstraintSurface
from geometric_sampling.manifold_sampling.utils import grad

from typing import Optional, Callable

import torch
import numpy as np
from scipy import stats, optimize

NEWTON_MAX_ITER = 50

class ManifoldMCMCSampler:

    def __init__(self, n_dim: int, scale: float, constraint_equation: Callable, metric: Optional[Callable]=None, density_function: Optional[Callable]=None) -> None:
        self.n_dim = n_dim
        self.surface = ConstraintSurface(n_dim, constraint_equation, metric)
        self.scale = scale

        if density_function:
            self.density_function = density_function
        else:
            self.density_function = lambda *args: 1.0

        
    def step(self, current_point):
        # Generate orthogonal basis for tangent plane and normal 
        tangent_space = self.surface.generate_tangent_space(current_point)
        normal_space = self.surface.generate_normal_space(current_point)

        # Sample tangent plane
        metric_tensor = self.surface.metric(current_point)
        covariance = torch.inverse(metric_tensor) * self.scale
        sample = np.random.multivariate_normal(mean=np.zeros(tangent_space.shape[1]), cov=covariance, size=1)
        v = tangent_space @ torch.tensor(sample).T
        p_v = stats.multivariate_normal.pdf(sample, mean=np.zeros(tangent_space.shape[1]), cov=covariance)

        # Solve for projected point 
        new_point = self._project(current_point + v, normal_space)

        # If projection fails, reject
        if new_point is None:
            return current_point

        # Find v' and p(v') for reverse projection step
        v_prime, p_v_prime = self._reverse_projection(current_point, new_point)

        # Check reverse projection works
        if v_prime is None:
            return current_point

        # Metropolis-Hastings rejection step
        u = np.random.random()
        acceptance_prob = (self.density_function(new_point) * p_v_prime) / (self.density_function(current_point) * p_v)
        if u > acceptance_prob:
            return current_point

        return new_point

    def sample(self, n_samples, initial_point=None, log_interval:Optional[int]=None):
        if initial_point is None:
            initial_point = self._get_initial_point()

        self.surface._check_constraint(initial_point)

        if not log_interval:
            log_interval = n_samples

        current_point = initial_point
        samples = [current_point]
        for i in range(n_samples):
            if i % log_interval == 0:
                print(f"step {i}")
            current_point = self.step(current_point)
            samples.append(current_point)

        return samples

    def _get_initial_point(self):
        raise NotImplementedError()

    
    def _reverse_projection(self, current_point: torch.Tensor, new_point: torch.Tensor):
        """Check if reverse projection new_point -> current_point is possible and is solved by Newton. 
        Returns v_prime reverse tangent vector and p_v_prime, probability of selecting that vector

        Args:
            current_point (torch.Tensor): starting point of mcmc step
            new_point (torch.Tensor): new point found after projection step
        """
        # Generate normal and tangent space for new point
        new_tangent_space = self.surface.generate_tangent_space(new_point)
        new_normal_space = self.surface.generate_normal_space(new_point)

        # find v_prime by projecting new_point - current_point onto tangent space
        dp = current_point - new_point
        dp_tangent = (dp.T @ new_tangent_space)
        v_prime =  new_tangent_space @ dp_tangent.T

        # check Newton solver converges
        if self._project(new_point + v_prime, new_normal_space) is None:
            return None, None

        metric_tensor = self.surface.metric(current_point)
        covariance = torch.inverse(metric_tensor) * self.scale
        p_v_prime = stats.multivariate_normal.pdf(dp_tangent, mean=np.zeros(new_tangent_space.shape[1]), cov=covariance)

        return v_prime, p_v_prime


    def _project(self, point: torch.Tensor, normal_space: torch.Tensor):
        """Compute projection of point to surface along normal using Newton's method.

        Args:
            point (torch.Tensor): starting point
            normal_space (torch.Tensor): normal space of surface corresponding to specific point
            n_iterations (int, optional): max number of iterations for Newton's method. Defaults to 50.

        Returns:
            (torch.Tensor, Optional): projected point, or None if not converged after n_iterations
        """

        projection_equation = lambda x : self.surface.constraint_equation(point + normal_space * x)
        projection_equation_grad = grad(projection_equation)
        root, r = optimize.newton(projection_equation, 0.0, fprime=projection_equation_grad, full_output=True, maxiter=NEWTON_MAX_ITER, disp=False)
        if r.converged:
            projection = point + normal_space * root
        else:
            projection = None

        return projection


if __name__ == "__main__":
    from IPython import embed
    embed()