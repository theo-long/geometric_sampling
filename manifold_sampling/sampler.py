from geometric_sampling.manifold_sampling.surfaces import ConstraintSurface
from geometric_sampling.manifold_sampling.utils import grad
from geometric_sampling.manifold_sampling.errors import ConstraintError

from typing import Optional, Callable, List

import torch
import numpy as np
from scipy import stats, optimize

NEWTON_MAX_ITER = 50
SPHERE_SOLVER_MAX_ITER = 250
DEFAULT_INTERPOLATING_PRECISION = 1e-4


class ManifoldMCMCSampler:
    def __init__(
        self,
        surface: ConstraintSurface,
        scale: float,
        density_function: Optional[Callable] = None,
        inequality_constraints: Optional[List[Callable]] = None,
    ) -> None:
        self.surface = surface
        self.scale = scale
        if inequality_constraints:
            self.inequality_constraints = inequality_constraints
        else:
            self.inequality_constraints = []

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
        covariance = np.linalg.inv(metric_tensor) * self.scale
        sample = np.random.multivariate_normal(
            mean=np.zeros(tangent_space.shape[0]), cov=covariance, size=1
        )
        v = (sample @ tangent_space).squeeze()
        p_v = stats.multivariate_normal.pdf(
            sample, mean=np.zeros(tangent_space.shape[0]), cov=covariance
        )

        # Solve for projected point
        new_point = self._project(current_point + v, normal_space)

        # If projection fails, reject
        if new_point is None:
            return current_point

        inequality_satisfaction = self._check_inequality_constraints(new_point)
        if not inequality_satisfaction:
            return current_point

        # Find v' and p(v') for reverse projection step
        v_prime, p_v_prime = self._reverse_projection(current_point, new_point)

        # Check reverse projection works
        if v_prime is None:
            return current_point

        # Metropolis-Hastings rejection step
        u = np.random.random()
        acceptance_prob = (self.density_function(new_point) * p_v_prime) / (
            self.density_function(current_point) * p_v
        )
        if u > acceptance_prob:
            return current_point

        return new_point

    def sample(self, n_samples, initial_point=None, log_interval: Optional[int] = None):
        if initial_point is None:
            initial_point = self._get_initial_point()

        self.surface._check_constraint(initial_point)
        if not self._check_inequality_constraints(initial_point):
            raise ConstraintError(
                "Inequality constraint not satisfied by starting point."
            )

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
        dtangent = new_tangent_space @ dp[:, None]

        v_prime = (dtangent.T @ new_tangent_space).squeeze()

        # check Newton solver converges to reverse point
        reverse_point = self._project(new_point + v_prime, new_normal_space)
        if reverse_point is None:
            return None, None
        if not np.allclose(current_point, reverse_point, atol=self.surface.tol):
            return None, None

        metric_tensor = self.surface.metric(current_point)
        covariance = np.linalg.inv(metric_tensor) * self.scale
        p_v_prime = stats.multivariate_normal.pdf(
            dtangent.squeeze(), mean=np.zeros(new_tangent_space.shape[0]), cov=covariance
        )

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

        projection_equation = lambda x: self.surface.constraint_equation(
            point + x @ normal_space
        ).squeeze()
        
        result = optimize.root(
            projection_equation,
            np.zeros(normal_space.shape[0]),
            method="hybr",
            options=dict(
                maxfev=50,
            )
        )
        if result.success:
            projection = point + result.x @ normal_space
        else:
            projection = None

        return projection

    def _check_inequality_constraints(self, point: torch.Tensor):
        for constraint in self.inequality_constraints:
            if constraint(point) <= 0.0:
                return False

        return True


class ManifoldSphereSampler:
    def __init__(
        self,
        surface:ConstraintSurface,
        inequality_constraints: Optional[List[Callable]] = None,
    ) -> None:
        self.surface = surface
        if inequality_constraints:
            self.inequality_constraints = inequality_constraints
        else:
            self.inequality_constraints = []

    def sample(
        self,
        n_samples: int,
        centre: torch.tensor,
        radius: float,
        log_interval: Optional[int] = None,
    ):
        if not log_interval:
            log_interval = n_samples

        centre = centre.squeeze()

        p1, p2 = self._get_sphere_points(n_samples, centre, radius)
        samples = []
        for i in range(n_samples):
            if i % log_interval == 0:
                print(f"step {i}")

            intersection_points = self._find_constraint_solutions(
                torch.tensor(p1[i]), torch.tensor(p2[i])
            )
            if intersection_points:
                samples.extend(intersection_points)

        return torch.stack(samples)

    def _find_constraint_solutions(
        self,
        p1: torch.Tensor,
        p2: torch.Tensor,
        precision=DEFAULT_INTERPOLATING_PRECISION,
    ):
        """Find solution to constraint on line p1-->p2, or return None if not found.

        Args:
            p1 (torch.Tensor): first point on sphere
            p2 (torch.Tensor): second point on sphere
        """
        # Find sign changes
        v = p2 - p1
        t_interpolating = torch.arange(0, 1 + precision, precision)
        interpolating_points = p1 + (t_interpolating * v[:, None]).T
        (interpolating_indices,) = np.where(
            np.diff(
                np.sign(
                    self.surface.constraint_equation(interpolating_points)
                ).flatten()
            )
            != 0
        )

        if len(interpolating_indices) == 0:
            return None

        def line_equation(t): 
            return self.surface.constraint_equation((p1 + v * t)[None, :])

        roots = []
        for t0 in t_interpolating[interpolating_indices]:
            solution = optimize.root_scalar(
                line_equation,
                bracket=(t0, t0 + precision),
                x0=t0,
                maxiter=SPHERE_SOLVER_MAX_ITER,
            )
            if solution.converged:
                roots.append(p1 + v * solution.root)

        if len(roots) == 0:
            return None

        if len(roots) < self.surface.n_dim - 1:
            print("missing root")
            
        return roots

    def _get_sphere_points(self, n_pairs: int, centre: torch.tensor, radius: float):
        gaussian_samples = np.random.multivariate_normal(
            mean=torch.zeros(self.surface.n_dim), cov=torch.eye(self.surface.n_dim), size=(n_pairs, 2)
        )
        sphere_samples = gaussian_samples / np.expand_dims(
            np.linalg.norm(gaussian_samples, axis=-1), -1
        )

        # scale and transform
        sphere_samples *= radius
        sphere_samples += np.expand_dims(centre.T, 0)

        return sphere_samples.transpose(1, 0, 2)


if __name__ == "__main__":

    def sphere_equation(p, R=1):
        return (np.linalg.norm(p, axis=1) - R)[:, None]

    sampler = ManifoldSphereSampler(3, constraint_equation=sphere_equation)
    points = sampler.sample(n_samples=10000, centre=torch.tensor([[0.],[0.],[0.]]), radius=2)

    from IPython import embed

    embed()
