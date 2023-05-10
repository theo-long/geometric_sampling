from geometric_sampling.manifold_sampling.surfaces import (
    ConstraintSurface,
    AlgebraicSurface,
)
from geometric_sampling.manifold_sampling.utils import (
    grad,
    sympy_func_to_array_func,
    change_affine_coordinates,
)
from geometric_sampling.manifold_sampling.solve import (
    generate_line_equation_coefficients,
    find_line_intersections,
    t,
    newton_solver,
    scipy_solver,
)
from geometric_sampling.manifold_sampling.errors import ConstraintError, RejectCode

from typing import Optional, Callable, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
from multiprocess import Pool
from tqdm import trange

import torch
import numpy as np
from scipy import stats
from numba import njit, float64

from logging import getLogger

log = getLogger()

SPHERE_SOLVER_MAX_ITER = 250
DEFAULT_INTERPOLATING_PRECISION = 1e-4

@dataclass
class StepInfo:
    nfev: int
    njev: int
    root: np.ndarray
    v: np.ndarray
    reject_code: RejectCode = RejectCode.NONE

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
        multiple_sol_every=None,
        projection_solver="newton",
        affine_coordinates=False,
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
            multiple_sol_every (int, optional): Use multiple solution sampling every this many steps. If None do not use.
            projection_solver (str): Which solver to use for projection step. Can be "newton" or "hybr" for scipy powell hybrid. Default is "newton"
        """
        self.surface = surface
        self.scale = scale
        self.use_jac = use_jac
        self.projection_solver = projection_solver
        self.affine_coordinates = affine_coordinates
        
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
            self.adaptive_scale = lambda point: (
                cov_matrix * min_scale
                + (1 - min_scale)
                * np.exp(-1 * (self.surface.mean_curvature(point) / alpha) ** 2)
            )
        else:
            raise ValueError(
                f"value {curvature_adaptive_scale} for curvature_adaptive_scale not recognized"
            )

        # pre-compute for speed
        self._tangent_space_mean = np.zeros(self.surface.n_dim - self.surface.codim)
        self._normal_space_mean = np.zeros(self.surface.codim)

        self.multiple_sol_every = multiple_sol_every
        if multiple_sol_every is not None:
            if self.surface.codim != 1:
                raise ValueError(
                    "Multiple solution sampling only implemented for codimension 1."
                )
            self.line_eq_coeffs = generate_line_equation_coefficients(self.surface)

    def _multiple_projection(self, point, normal_space, return_all=False):
        # Solve for projected point
        line_eq_coeffs = self.line_eq_coeffs(point, point + normal_space[0])
        poly = np.polynomial.Polynomial(line_eq_coeffs[::-1])
        roots = poly.roots()
        real_roots = np.real(roots[(np.abs(np.imag(roots)) < 1e-14)])

        if len(real_roots) == 0:
            return None

        # Sort by distance and choose according to weighting
        real_roots = real_roots[np.argsort(np.abs(real_roots))]
        p = self.solution_weights[: real_roots.shape[0]]
        p = p / p.sum()

        if return_all:
            return real_roots, p

        root = np.random.choice(real_roots, p=p)
        new_point = point + root * normal_space[0]
        w = p[real_roots == root]
        return new_point, w

    def _reverse_multiple_projection(self, current_point, new_point):
        # Find v' and p(v') for reverse projection step
        (
            new_tangent_space,
            new_normal_space,
        ) = self.surface.generate_tangent_and_normal_space(new_point)

        # find v_prime by projecting new_point - current_point onto tangent space
        dp = current_point - new_point
        dtangent = new_tangent_space @ dp[:, None]

        v_prime = (dtangent.T @ new_tangent_space).squeeze()

        points, w_primes = self._multiple_projection(
            new_point + v_prime, new_normal_space
        )
        index = (points - new_point).argmin()
        return v_prime, w_primes[index]

    def multiple_solution_step(self, current_point):
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
            sample, mean=self._tangent_space_mean, cov=covariance
        )

        new_point, w = self._multiple_projection(current_point + v, normal_space)

        # If projection fails, reject
        if new_point is None:
            return current_point, RejectCode.PROJECTION, 0

        inequality_satisfaction = self._check_inequality_constraints(new_point)
        if not inequality_satisfaction:
            return current_point, RejectCode.INEQUALITY, 0

        p_v_prime, w_prime = self._reverse_multiple_projection(current_point, new_point)

        # Metropolis-Hastings rejection step
        u = np.random.random()
        acceptance_prob = (self.density_function(new_point) * p_v_prime * w_prime) / (
            self.density_function(current_point) * p_v * w
        )
        if u > acceptance_prob:
            return current_point, RejectCode.MH, 0

        return new_point, RejectCode.NONE, 0

    def train(self, n_train: int, initial_point: np.ndarray, verbose=True):
        """Generate an index of points for fast newton initialization.

        Args:
            n_train (int): number of samples for index
            initial_point (np.ndarray): initial point for sampling start
        """

        index_points, rej, _ = self.sample(n_train, initial_point=initial_point, verbose=verbose)
        index_points = np.ascontiguousarray(index_points[rej == 0])

        @njit(float64[::1](float64[::1], float64[:, ::1], float64[:, ::1]))
        def initial_root(point, tangent_space, normal_space):
            projection_vectors = (index_points - point)
            index = ((projection_vectors @ tangent_space.T) ** 2).sum(axis=1).argmin()
            root = projection_vectors[index] @ normal_space.T / (normal_space ** 2).sum(axis=1)
            return root

        self.generate_root = initial_root

    def step(self, current_point, root_prediction):

        # Generate orthogonal basis for tangent plane and normal
        tangent_space, normal_space = self.surface.generate_tangent_and_normal_space(
            current_point
        )

        # Sample tangent plane
        covariance = self.adaptive_scale(current_point)
        sample = np.random.multivariate_normal(
            mean=self._tangent_space_mean, cov=covariance, size=1
        )
        p_v = stats.multivariate_normal.pdf(
            sample, mean=self._tangent_space_mean, cov=covariance
        )
        v = (sample @ tangent_space).squeeze()
        # Solve for projected point
        new_point, nfev, njev, root = self._project(current_point + v, tangent_space, normal_space, root_prediction)
        step_info = StepInfo(nfev, njev, root, sample)

        # If projection fails, reject
        if new_point is None:
            step_info.reject_code = RejectCode.PROJECTION
            return current_point, step_info

        inequality_satisfaction = self._check_inequality_constraints(new_point)
        if not inequality_satisfaction:
            step_info.reject_code = RejectCode.INEQUALITY
            return current_point, step_info

        # Generate normal and tangent space for new point
        (
            new_tangent_space,
            new_normal_space,
        ) = self.surface.generate_tangent_and_normal_space(new_point)

        # find v_prime by projecting new_point - current_point onto tangent space
        dp = current_point - new_point
        dtangent = new_tangent_space @ dp[:, None]
        v_prime = (dtangent.T @ new_tangent_space).squeeze()

        # calculate probability of new point -> current transition
        covariance = self.adaptive_scale(new_point)
        p_v_prime = stats.multivariate_normal.pdf(
            dtangent.squeeze(),
            mean=self._tangent_space_mean,
            cov=covariance,
        )

        # Metropolis-Hastings rejection step
        u = np.random.random()
        prob_ratio = (self.density_function(new_point) * p_v_prime) / (
            self.density_function(current_point) * p_v
        )
        # detJ_ratio = (normal_space @ normal_space.T) / (new_normal_space @ new_normal_space.T)
        acceptance_prob = prob_ratio
        if u > acceptance_prob:
            step_info.reject_code = RejectCode.MH
            return current_point, step_info

        # reverse projection step
        if not self._reverse_projection(
            current_point, new_point, new_tangent_space, new_normal_space, v_prime, root_prediction,
        ):
            step_info.reject_code = RejectCode.REVERSE_PROJECTION
            return current_point, step_info

        return new_point, step_info

    def sample(self, n_samples: int, initial_point:Optional[np.ndarray]=None, root_prediction:bool=False,  verbose:bool=True, log_info:bool=False):
        """Sample points with MCMC.

        Args:
            n_samples (int): _description_
            initial_point (Optional[np.ndarray], optional): Initial point to start sampling. Defaults to None.
            root_prediction (bool, optional): Whether to use root prediction. Defaults to False.
            verbose (bool, optional): Logging progress. Defaults to True.
            log_info (bool, optional): Extra info like number of function evals. Defaults to False.

        Returns:
            points
        """
        
        if initial_point is None:
            initial_point = self._get_initial_point()
        else:
            initial_point = initial_point.astype(np.float64)

        if root_prediction is True and not hasattr(self, "generate_root"):
            raise ValueError("Must call train method first to use root prediction.")

        self.surface._check_constraint(initial_point)
        if not self._check_inequality_constraints(initial_point):
            raise ConstraintError(
                "Inequality constraint not satisfied by starting point."
            )

        current_point = initial_point
        samples = np.zeros((n_samples, self.surface.n_dim), dtype=np.float64)
        samples[0] = current_point
        reject_codes = np.zeros(n_samples)
        reject_codes[0] = RejectCode.NONE

        if log_info:
            nfev = np.zeros(n_samples)
            njev = np.zeros(n_samples)
            roots = np.zeros((n_samples, self.surface.codim))
            proposals = np.zeros((n_samples, self.surface.n_dim - self.surface.codim))
            info = {
                "nfev":nfev,
                "njev":njev,
                "roots":roots,
                "proposals":proposals,
            }
        else:
            info = {}


        multiple_sol_every = (
            self.multiple_sol_every
            if self.multiple_sol_every is not None
            else n_samples
        )
        iterable = trange(1, n_samples) if verbose else range(1, n_samples)
        for i in iterable:
            if i % multiple_sol_every == 0 and i > 0:
                current_point, reject_code, nfev = self.multiple_solution_step(
                    current_point
                )
            else:
                current_point, step_info = self.step(current_point, root_prediction)

            if self.affine_coordinates:
                current_point = change_affine_coordinates(current_point)

            samples[i] = current_point
            reject_codes[i] = step_info.reject_code
        
            if log_info:
                nfev[i] = step_info.nfev
                njev[i] = step_info.njev
                roots[i] = step_info.root
                proposals[i] = step_info.v


        return samples, reject_codes, info

    def _get_initial_point(self):
        raise NotImplementedError()

    def _reverse_projection(
        self,
        current_point: np.ndarray,
        new_point: np.ndarray,
        new_tangent_space: np.ndarray,
        new_normal_space: np.ndarray,
        v_prime: np.ndarray,
        root_prediction: bool,
    ):
        """Check if reverse projection new_point -> current_point is possible and is solved by Newton.
        Returns v_prime reverse tangent vector and p_v_prime, probability of selecting that vector

        Args:
            current_point (np.ndarray): starting point of mcmc step
            new_point (np.ndarray): new point found after projection step
            new_normal_space (np.ndarray)
            v_prime (np.ndarray)
        """

        # check Newton solver converges to reverse point
        reverse_point, _, _, _ = self._project(new_point + v_prime, new_tangent_space, new_normal_space, root_prediction)
        if reverse_point is None:
            return False
        if not np.allclose(current_point, reverse_point, atol=self.surface.tol):
            return False

        return True

    def _project(self, point: np.ndarray, tangent_space: np.ndarray, normal_space: np.ndarray, root_prediction):
        """Compute projection of point to surface along normal using Newton's method.

        Args:
            point (np.ndarray): starting point
            tangent_space (np.ndarray): tangent space of surface corresponding to specific point
            normal_space (np.ndarray): normal space of surface corresponding to specific point
            n_iterations (int, optional): max number of iterations for Newton's method. Defaults to 50.

        Returns:
            projection, nfev, njev, root
        """

        def projection_equation(x):
            return self.surface.jitted_constraint_equation(
                point + x @ normal_space
            ).squeeze(1)

        if self.use_jac:
            projected_jacobian = (
                lambda x: self.surface.jitted_jacobian(point + x @ normal_space)
                @ normal_space.T
            )
        else:
            projected_jacobian = None

        if root_prediction:
            x0 = self.generate_root(point, tangent_space, normal_space)
        else:
            x0 = self._normal_space_mean

        if self.projection_solver == "newton":
            root, nfev, njev = newton_solver(
                F=projection_equation,
                J=projected_jacobian,
                x=x0,
                eps=1.49012e-08,
            )
        else:
            root, nfev, njev = scipy_solver(
                F=projection_equation,
                J=projected_jacobian,
                x=x0,
                eps=1.49012e-08,
                method=self.projection_solver,
            )

        if root is np.NaN:
            projection = None
        else:
            projection = point + root @ normal_space

        return projection, nfev, njev, root

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

        p1, p2 = get_sphere_points(self.surface.n_dim, n_samples, centre, radius)
        samples = []
        reject_codes = []
        for i in trange(n_samples):
            p, q = p1[i], p2[i]
            coeffs = self.line_eq_coeffs(np.concatenate([p, q]))
            intersection_points = find_line_intersections(coeffs, p, q, return_all=True)
            if intersection_points is not None:
                samples.append(intersection_points)
                reject_codes.append(RejectCode.NONE)
            else:
                reject_codes.append(RejectCode.PROJECTION)

        return np.concatenate(samples), reject_codes


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
                lambda point: np.linalg.inv(self.surface.metric(point)) * self.scale
            )
        elif curvature_adaptive_scale == "mean_curvature":
            self.adaptive_scale = (
                lambda point: np.linalg.inv(self.surface.metric(point))
                * self.scale
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
        sphere_points = get_sphere_points(
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


def get_sphere_points(n_dim, n_pairs: int, centre: torch.tensor, radius: float):
    gaussian_samples = np.random.multivariate_normal(
        mean=np.zeros(n_dim),
        cov=np.eye(n_dim),
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
