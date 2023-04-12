from geometric_sampling.manifold_sampling.surfaces import (
    ConstraintSurface,
    AlgebraicSurface,
)
from geometric_sampling.manifold_sampling.utils import grad, sympy_func_to_array_func
from geometric_sampling.manifold_sampling.solve import generate_line_equation_coefficients, find_line_intersections, t
from geometric_sampling.manifold_sampling.errors import ConstraintError, RejectCode

from typing import Optional, Callable, List
from abc import ABC, abstractmethod
from multiprocessing import Pool
from tqdm import trange

import torch
import numpy as np
from scipy import stats, optimize
import sympy

from logging import getLogger

log = getLogger()

NEWTON_MAX_ITER = 50
SPHERE_SOLVER_MAX_ITER = 250
DEFAULT_INTERPOLATING_PRECISION = 1e-4

class Sampler(ABC):
    @abstractmethod
    def __init__(
        self,
        surface: ConstraintSurface,
        *args,
        density_function: Optional[Callable] = None,
        inequality_constraints: Optional[List[Callable]] = None,
        **kwargs,
    ) -> None:
        pass

    @abstractmethod
    def sample(self, n_samples: int) -> List:
        pass


class ManifoldMCMCSampler(Sampler):
    def __init__(
        self,
        surface: ConstraintSurface,
        scale: float,
        density_function: Optional[Callable] = None,
        inequality_constraints: Optional[List[Callable]] = None,
        curvature_adaptive_scale: Optional[str] = None,
        alpha=1.0,
        min_scale=0.1,
        use_jac=True,
    ) -> None:
        """An MCMC sampler for constraint manifolds based on Zappa et al. (2017)

        Args:
            surface (ConstraintSurface): The surface to sample from
            scale (float): length scale for proposal gaussian
            density_function (Optional[Callable], optional): The unnormalized pdf on the surface. If None defaults to uniform. Defaults to None.
            inequality_constraints (Optional[List[Callable]], optional): Additional inequality constraints for sampler. Defaults to None.
            curvature_adaptive_scale (Optional[str], optional): Method to use for curvature adaptive proposals. Defaults to None.
            alpha (float, optional): alpha parameter for curvature adaptation. Defaults to 1.0.
            min_scale (float, optional): minimum scale parameter for curvature adaptation. Defaults to 0.1.
            use_jac (bool, optional): Whether or not to use gradients for projection steps. Defaults to True.
        """
        self.surface = surface
        self.scale = scale
        self.use_jac = use_jac
        if inequality_constraints:
            self.inequality_constraints = inequality_constraints
        else:
            self.inequality_constraints = []

        if density_function:
            self.density_function = density_function
        else:
            self.density_function = lambda *args: 1.0

        cov_matrix = np.eye(self.surface.n_dim - self.surface.codim) * self.scale
        if curvature_adaptive_scale is None:          
            self.adaptive_scale = lambda point: cov_matrix
        elif curvature_adaptive_scale == "mean_curvature":
            self.adaptive_scale = (
                lambda point:
                (
                    cov_matrix
                    * min_scale
                    + (1 - min_scale)
                    * np.exp(-1 * (self.surface.mean_curvature(point) / alpha) ** 2)
                )
            )
        else:
            raise ValueError(
                f"value {curvature_adaptive_scale} for curvature_adaptive_scale not recognized"
            )

        # pre-compute for speed
        self._tangent_space_mean = np.zeros(self.surface.n_dim - self.surface.codim)
        self._normal_space_mean = np.zeros(self.surface.codim)

    def step(self, current_point):
        # Generate orthogonal basis for tangent plane and normal
        tangent_space, normal_space = self.surface.generate_tangent_and_normal_space(
            current_point
        )

        # Sample tangent plane
        covariance = self.adaptive_scale(current_point)
        sample = np.random.multivariate_normal(
            mean=self._tangent_space_mean, cov=covariance, size=1
        )
        v = (sample @ tangent_space).squeeze()
        p_v = stats.multivariate_normal.pdf(
            sample,  mean=self._tangent_space_mean, cov=covariance
        )

        # Solve for projected point
        new_point, nfev = self._project(current_point + v, normal_space)

        # If projection fails, reject
        if new_point is None:
            return current_point, RejectCode.PROJECTION, nfev

        inequality_satisfaction = self._check_inequality_constraints(new_point)
        if not inequality_satisfaction:
            return current_point, RejectCode.INEQUALITY, nfev

        # Find v' and p(v') for reverse projection step
        v_prime, p_v_prime = self._reverse_projection(current_point, new_point)

        # Check reverse projection works
        if v_prime is None:
            return current_point, RejectCode.REVERSE_PROJECTION, nfev

        # Metropolis-Hastings rejection step
        u = np.random.random()
        acceptance_prob = (self.density_function(new_point) * p_v_prime) / (
            self.density_function(current_point) * p_v
        )
        if u > acceptance_prob:
            return current_point, RejectCode.MH, nfev

        return new_point, RejectCode.NONE, nfev

    def sample(self, n_samples, initial_point=None):
        if initial_point is None:
            initial_point = self._get_initial_point()

        self.surface._check_constraint(initial_point)
        if not self._check_inequality_constraints(initial_point):
            raise ConstraintError(
                "Inequality constraint not satisfied by starting point."
            )

        current_point = initial_point
        samples = np.zeros((n_samples + 1, self.surface.n_dim))
        samples[0] = current_point
        reject_codes = np.zeros(n_samples + 1)
        reject_codes[0] = RejectCode.NONE
        project_steps = np.zeros(n_samples + 1)
        for i in trange(n_samples):
            current_point, reject_code, nfev = self.step(current_point)
            samples[i] = current_point
            reject_codes[i] = reject_code
            project_steps[i] = nfev

        return np.stack(samples), reject_codes, project_steps

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
        (
            new_tangent_space,
            new_normal_space,
        ) = self.surface.generate_tangent_and_normal_space(new_point)

        # find v_prime by projecting new_point - current_point onto tangent space
        dp = current_point - new_point
        dtangent = new_tangent_space @ dp[:, None]

        v_prime = (dtangent.T @ new_tangent_space).squeeze()

        # check Newton solver converges to reverse point
        reverse_point, _ = self._project(new_point + v_prime, new_normal_space)
        if reverse_point is None:
            return None, None
        if not np.allclose(current_point, reverse_point, atol=self.surface.tol):
            return None, None

        covariance = self.adaptive_scale(new_point)
        p_v_prime = stats.multivariate_normal.pdf(
            dtangent.squeeze(),
            mean=self._tangent_space_mean,
            cov=covariance,
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

        if self.use_jac:
            projected_jacobian = lambda x: self.surface.jacobian(point + x @ normal_space) @ normal_space.T
        else:
            projected_jacobian = None

        result = optimize.root(
            projection_equation,
            self._normal_space_mean,
            method="hybr",
            options=dict(
                maxfev=NEWTON_MAX_ITER,
            ),
            jac=projected_jacobian,
        )
        if result.success:
            projection = point + result.x @ normal_space
        else:
            projection = None

        return projection, result.nfev

    def _check_inequality_constraints(self, point: torch.Tensor):
        for constraint in self.inequality_constraints:
            if constraint(point) <= 0.0:
                return False

        return True


class ManifoldSphereSampler(Sampler):
    def __init__(
        self,
        surface: ConstraintSurface,
        density_function: Optional[Callable] = None,
        inequality_constraints: Optional[List[Callable]] = None,
    ) -> None:
        self.surface = surface
        if inequality_constraints:
            self.inequality_constraints = inequality_constraints
        else:
            self.inequality_constraints = []

        if density_function:
            self.density_function = density_function
        else:
            self.density_function = lambda *args: 1.0

        # Precompute line equations
        self.line_eq_coeffs = generate_line_equation_coefficients(self.surface)

    def sample(
        self,
        n_samples: int,
        centre: torch.tensor,
        radius: float,
        precision=DEFAULT_INTERPOLATING_PRECISION,
    ):

        centre = centre.squeeze()

        p1, p2 = _get_sphere_points(self.surface.n_dim, n_samples, centre, radius)
        samples = []
        reject_codes = []
        for i in trange(n_samples):
            p, q = p1[i], p2[i]
            coeffs = self.line_eq_coeffs(np.concatenate([p, q]))
            intersection_point = find_line_intersections(coeffs, p, q)
            if intersection_point is not None:
                samples.append(intersection_point)
                reject_codes.append(RejectCode.NONE)
            else:
                reject_codes.append(RejectCode.PROJECTION)

        return np.stack(samples), reject_codes


class ManifoldSphereMCMCSampler(Sampler):
    def __init__(
        self,
        surface: ConstraintSurface,
        scale: float,
        density_function: Optional[Callable] = None,
        inequality_constraints: Optional[List[Callable]] = None,
        curvature_adaptive_scale: Optional[str] = None,
        alpha=1.0,
        min_scale=0.1,
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

        if curvature_adaptive_scale is None:
            self.adaptive_scale = (
                lambda point: np.linalg.inv(self.surface.metric(point)) * self.scale ** 2
            )
        elif curvature_adaptive_scale == "mean_curvature":
            self.adaptive_scale = (
                lambda point: np.linalg.inv(self.surface.metric(point))
                * self.scale ** 2
                * (
                    min_scale
                    + (1 - min_scale)
                    * np.exp(-1 * (self.surface.mean_curvature(point) / alpha) ** 2)
                )
            )
        else:
            raise ValueError(
                f"value {curvature_adaptive_scale} for curvature_adaptive_scale not recognized"
            )

        # Precompute line equations
        self.line_eq_coeffs = generate_line_equation_coefficients(self.surface)

    def step(self, current_point, sphere_pair):
        scale = self.adaptive_scale(current_point)
        sphere_pair = (scale @ sphere_pair) + current_point

        coeffs = self.line_eq_coeffs(sphere_pair.flatten())
        new_point = find_line_intersections(coeffs, sphere_pair[0], sphere_pair[1])

        if new_point is None:
            return current_point, RejectCode.PROJECTION

        # Check constraint
        inequality_satisfaction = self._check_inequality_constraints(new_point)
        if not inequality_satisfaction:
            return current_point, RejectCode.INEQUALITY

        # MH accept/reject
        u = np.random.random()
        acceptance_prob = self.density_function(new_point) / self.density_function(
            current_point
        )
        if u > acceptance_prob:
            return current_point, RejectCode.MH

        return new_point, RejectCode.NONE

    def sample(self, n_samples, initial_point):

        self.surface._check_constraint(initial_point)
        if not self._check_inequality_constraints(initial_point):
            raise ConstraintError(
                "Inequality constraint not satisfied by starting point."
            )

        current_point = initial_point
        samples = [current_point]
        reject_codes = [RejectCode.NONE]
        sphere_points = _get_sphere_points(
            self.surface.n_dim,
            n_pairs=n_samples,
            centre=np.zeros(self.surface.n_dim),
            radius=1,
        )

        for i in trange(n_samples):
            sphere_pair = sphere_points[:, i, :]
            current_point, reject_code = self.step(current_point, sphere_pair)
            samples.append(current_point)
            reject_codes.append(reject_code)

        return np.stack(samples), reject_codes

    def _check_inequality_constraints(self, point: torch.Tensor):
        for constraint in self.inequality_constraints:
            if constraint(point) <= 0.0:
                return False

        return True


class LinearSubspaceSampler(Sampler):
    def __init__(
        self,
        surface: AlgebraicSurface,
        density_function: Optional[Callable] = None,
        inequality_constraints: Optional[List[Callable]] = None,
    ) -> None:

        self.surface = surface
        if inequality_constraints:
            self.inequality_constraints = inequality_constraints
        else:
            self.inequality_constraints = []

        if density_function:
            self.density_function = density_function
        else:
            self.density_function = lambda *args: 1.0

    def sample(self, n_samples: int, log_interval: int) -> List:

        if not log_interval:
            log_interval = n_samples

        samples = []
        for i in range(n_samples):
            if i % log_interval == 0:
                print(f"step {i}")

            samples.extend(points)

        return points

    def polynomial_solve(self):
        pass

    def alpha(self):
        pass

    def random_subspace(self):
        pass


def _get_sphere_points(n_dim, n_pairs: int, centre: torch.tensor, radius: float):
    gaussian_samples = np.random.multivariate_normal(
        mean=torch.zeros(n_dim),
        cov=torch.eye(n_dim),
        size=(n_pairs, 2),
    )
    sphere_samples = gaussian_samples / np.expand_dims(
        np.linalg.norm(gaussian_samples, axis=-1), -1
    )

    # scale and transform
    sphere_samples *= radius
    sphere_samples += np.expand_dims(centre.T, 0)

    return sphere_samples.transpose(1, 0, 2)


def multiprocess_samples(
    surface_factory,
    sampler_factory,
    n_samples,
    initial_points,
    scale,
    n_procs,
    **surface_kwargs,
):
    pool = Pool(n_procs)
    args = [(n_samples, initial_point) for initial_point in initial_points]

    def sample(n_samples, initial_point):
        surface = surface_factory(**surface_kwargs)
        sampler = sampler_factory(surface=surface, scale=scale)
        return sampler.sample(n_samples=n_samples, initial_point=initial_point)

    return pool.starmap(sample, args)


if __name__ == "__main__":

    def sphere_equation(p, R=1):
        return (np.linalg.norm(p, axis=1) - R)[:, None]

    sampler = ManifoldSphereSampler(3, constraint_equation=sphere_equation)
    points = sampler.sample(
        n_samples=10000, centre=torch.tensor([[0.0], [0.0], [0.0]]), radius=2
    )

    from IPython import embed

    embed()
