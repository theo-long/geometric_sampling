from geometric_sampling.manifold_sampling.errors import ConstraintError
from geometric_sampling.manifold_sampling.utils import sympy_func_to_array_func
from typing import Callable, Optional, List, Union
import torch
import numpy as np
import scipy
import sympy
from sympy import symbols

x0, x1, x2 = symbols("x:3")

DEFAULT_TOLERANCE = 1e-12


class ConstraintSurface:
    def __init__(
        self,
        n_dim: int,
        constraint_equation: Callable[[torch.Tensor], torch.Tensor],
        metric: Optional[Callable] = None,
        tol=DEFAULT_TOLERANCE,
    ) -> None:
        self.n_dim = n_dim
        self.constraint_equation = constraint_equation
        if metric:
            self.metric = metric
        else:
            out = constraint_equation(torch.zeros(n_dim))
            self.metric = _euclidean_metric(n_dim - out.shape[1])
        self.tol = tol

    def generate_tangent_space(self, x: torch.Tensor):
        """Generates the tangent space at a point x, given as an orthnormal basis."""
        normal = self.generate_normal_space(x)
        tangent = scipy.linalg.null_space(normal).T
        return tangent

    def generate_normal_space(self, x: torch.Tensor):
        """Generates the subspace normal to the tangent space at x, given as an orthonormal basis."""
        x = x.requires_grad_()
        constraint_value = self._check_constraint(x).sum()
        constraint_value.backward()
        result = x.grad
        x = x.requires_grad_(False)
        return result

    def _check_constraint(self, x):
        constraint_value = self.constraint_equation(x)
        if isinstance(constraint_value, torch.Tensor):
            constraint_value = constraint_value.detach()
        if not np.allclose(constraint_value, 0, self.tol):
            raise ConstraintError(
                f"Point does not satisfy constraint equation with tolerance {self.tol:.2E}"
            )
        return constraint_value

    def __call__(self, *args) -> torch.Tensor:
        return self.constraint_equation(*args)


def symbolic_jacobian(f: sympy.Matrix, args):
    return f.jacobian(args)


def symbolic_hessian(f: sympy.Matrix, args):
    return sympy.hessian(f, args)


class AlgebraicSurface(ConstraintSurface):
    def __init__(
        self,
        n_dim: int,
        constraint_equations: Union[List[sympy.Poly], sympy.Poly],
        metric: Optional[Callable] = None,
        tol=DEFAULT_TOLERANCE,
    ) -> None:
        self.n_dim = n_dim
        self.args = symbols(f"x:{n_dim}")
        # convert to expressions from poly
        if type(constraint_equations) is not list:
            constraint_equations = [constraint_equations.expr]
        else:
            constraint_equations = [eq.expr for eq in constraint_equations]

        self.constraint_equation = sympy_func_to_array_func(
            self.args, sympy.Matrix(constraint_equations)
        )
        self.algebraic_equation = sympy.Matrix(constraint_equations)
        if metric:
            self.metric = metric
        else:
            self.metric = _euclidean_metric(n_dim - len(constraint_equations))
        self.tol = tol

    def generate_normal_space(self, x: torch.Tensor):
        """Generates the subspace normal to the tangent space at x, given as an orthonormal basis."""
        jac = self.jacobian(x)
        return jac

    def n_intersections(self, p1: torch.Tensor, p2: torch.Tensor):
        """Find number of intersection points on line between p1 and p2

        Args:
            p1 (torch.Tensor): first endpoint
            p2 (torch.Tensor): second endpoint
        """
        raise NotImplementedError()

    def jacobian(self, p1: torch.Tensor):
        jacobian_func = getattr(self, "_jacobian_func", None)
        if jacobian_func is None:
            jacobian = symbolic_jacobian(self.algebraic_equation, self.args)
            self._jacobian_func = sympy_func_to_array_func(self.args, jacobian)
            return self._jacobian_func(p1)
        else:
            return jacobian_func(p1)

    def hessian(self, p1: torch.Tensor):
        hessian_func = getattr(self, "_hessian_func", None)
        if hessian_func is None:
            hessian = symbolic_hessian(self.algebraic_equation, self.args)
            self._hessian_func = sympy_func_to_array_func(self.args, hessian)
            return self._hessian_func(p1)
        else:
            return hessian_func(p1)


class Torus(AlgebraicSurface):
    def __init__(
        self, r, R, metric: Optional[Callable] = None, tol=DEFAULT_TOLERANCE
    ) -> None:
        constraint_equation = (
            x0**2 + x1**2 + x2**2 + R**2 - r**2
        ) ** 2 - 4 * (R**2) * (x0**2 + x1**2)
        constraint_equation = sympy.Poly(constraint_equation)
        super().__init__(
            n_dim=3, constraint_equations=constraint_equation, metric=metric, tol=tol
        )


class Sphere(AlgebraicSurface):
    def __init__(
        self, r, n_dim, metric: Optional[Callable] = None, tol=DEFAULT_TOLERANCE
    ) -> None:
        variables = symbols(f"x:{n_dim}")
        constraint_equation = sympy.Poly(
            sympy.Add(*[v**2 for v in variables]) - r**2
        )
        super().__init__(
            n_dim=n_dim,
            constraint_equations=constraint_equation,
            metric=metric,
            tol=tol,
        )


class Peanut(AlgebraicSurface):
    def __init__(
        self, metric: Optional[Callable] = None, tol=DEFAULT_TOLERANCE
    ) -> None:
        constraint_equation = sympy.Poly(
            ((x0 + 1) ** 2 + x1**2 + x2**2)(
                (x0 - 1) ** 2 + 0.05 * x1**2 + 0.05 * x2**2
            )
            - 1.001
        )
        super().__init__(
            n_dim=3, constraint_equations=constraint_equation, metric=metric, tol=tol
        )


class TriFold(AlgebraicSurface):
    def __init__(
        self, power: int, metric: Optional[Callable] = None, tol=DEFAULT_TOLERANCE
    ) -> None:
        if power % 2 == 0:
            raise ValueError("Power must be odd")
        constraint_equation = sympy.Poly(x0**power + x1**power + x2**power)
        super().__init__(
            n_dim=3, constraint_equations=constraint_equation, metric=metric, tol=tol
        )


class RoundedCube(AlgebraicSurface):
    def __init__(
        self, r, power, metric: Optional[Callable] = None, tol=DEFAULT_TOLERANCE
    ) -> None:
        """Makes sphere more and more 'cube-like' as power increases."""
        if power % 2 != 0:
            raise ValueError("Power must be a multiple of 2")

        constraint_equation = sympy.Poly(
            x0**power + x1**power + x2**power - r**2
        )
        super().__init__(
            n_dim=3, constraint_equations=constraint_equation, metric=metric, tol=tol
        )


class Smartie(AlgebraicSurface):
    def __init__(
        self, z_factor: float, metric: Optional[Callable] = None, tol=DEFAULT_TOLERANCE
    ) -> None:
        """Flattened sphere shape."""
        constraint_equation = sympy.Poly(x0**2 + x1**2 + z_factor * x2**2)
        super().__init__(
            n_dim=3, constraint_equations=constraint_equation, metric=metric, tol=tol
        )


class SimpleAlgebraicIntersection(AlgebraicSurface):
    def __init__(
        self, metric: Optional[Callable] = None, tol=DEFAULT_TOLERANCE
    ) -> None:
        eq1 = sympy.Poly(x0**4 + x1**3 + x1 * x1 + x0**2 * x2 - x2 - 1)
        eq2 = sympy.Poly(x2 * x1 - x1 * x1 - 1)
        super().__init__(
            n_dim=3, constraint_equations=[eq1, eq2], metric=metric, tol=tol
        )


def _euclidean_metric(n_dim):
    return lambda x : np.eye(n_dim)
