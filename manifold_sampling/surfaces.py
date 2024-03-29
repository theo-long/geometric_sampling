from geometric_sampling.manifold_sampling.errors import ConstraintError
from geometric_sampling.manifold_sampling.utils import (
    sympy_func_to_array_func,
    sympy_func_to_jit_func,
)
from typing import Callable, Optional, List, Union, Iterable
import torch
import numpy as np
import scipy
import sympy
from sympy import symbols
import cyjax

x0, x1, x2, x3 = symbols("x:4")

DEFAULT_TOLERANCE = 1e-12

def _euclidean_metric(n_dim):
    return lambda x: np.eye(n_dim)
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
        self.jitted_constraint_equation = constraint_equation
        self.complex = False
        if metric:
            self.metric = metric
        else:
            out = constraint_equation(torch.zeros(n_dim))
            self.metric = _euclidean_metric(n_dim - out.shape[0])
            self.codim = out.shape[0]
        self.tol = tol

    def mean_curvature(self, x: np.array):
        raise NotImplementedError()

    def generate_tangent_space(self, x: torch.Tensor):
        """Generates the tangent space at a point x, given as an orthonormal basis."""
        normal = self.generate_normal_space(x)
        Q, _ = np.linalg.qr(normal.T, mode="complete")
        return Q[:, -self.n_dim + self.codim :].T

    def generate_normal_space(self, x: torch.Tensor):
        """Generates the subspace normal to the tangent space at x, given as an orthonormal basis."""
        x = x.requires_grad_()
        constraint_value = self._check_constraint(x).sum()
        constraint_value.backward()
        result = x.grad
        x = x.requires_grad_(False)
        return result

    def generate_tangent_and_normal_space(self, x: torch.Tensor):
        """Generate subspaces normal and tangent to manifold at x, as well as orthonormal version of normal space."""
        normal = self.generate_normal_space(x)
        Q, _ = np.linalg.qr(normal.T, mode="complete")
        n_hat, tangent_space = np.array_split(Q.T, [-self.n_dim + self.codim])
        return np.ascontiguousarray(tangent_space), normal, np.ascontiguousarray(n_hat)

    def _check_constraint(self, x):
        constraint_value = self.constraint_equation(x)
        if isinstance(constraint_value, torch.Tensor):
            constraint_value = constraint_value.detach()
        if not np.allclose(constraint_value, 0, atol=self.tol):
            raise ConstraintError(
                f"Point does not satisfy constraint equation with tolerance {self.tol:.2E}"
            )
        return constraint_value

    def __call__(self, *args) -> torch.Tensor:
        return self.constraint_equation(*args)


def symbolic_jacobian(f: sympy.Matrix, args) -> sympy.Matrix:
    return f.jacobian(args)


def symbolic_hessian(f: sympy.Matrix, args) -> sympy.Matrix:
    return sympy.hessian(f, args)


def symbolic_mean_curvature(f: sympy.Matrix, args) -> sympy.Matrix:
    jac = symbolic_jacobian(f, args)
    hessian = symbolic_hessian(f, args)
    return (
        (jac @ hessian @ jac.T)[0] - jac.norm() ** 2 * hessian.trace()
    ) / jac.norm() ** 3


def symbolic_gaussian_curvature(f: sympy.Matrix, args) -> sympy.Matrix:
    jac = symbolic_jacobian(f, args)
    hessian = symbolic_hessian(f, args)
    return -(
        sympy.Matrix([[hessian, jac.T], [jac, sympy.zeros(1, 1)]]).det()
        / (jac.norm() ** 4)
    )


def symbolic_shape_operator(f: sympy.Matrix, args) -> sympy.Matrix:
    jac = symbolic_jacobian(f, args)
    hessian = symbolic_hessian(f, args)
    normal = jac / jac.norm()
    return (sympy.eye(hessian.shape[0]) - normal.T @ normal) @ (hessian / jac.norm())


class AlgebraicSurface(ConstraintSurface):
    def __init__(
        self,
        n_dim: int,
        constraint_equations: Union[List[sympy.Poly], sympy.Poly],
        args: Optional[Iterable[sympy.Symbol]] = None,
        metric: Optional[Callable] = None,
        complex: bool = False,
        tol=DEFAULT_TOLERANCE,
    ) -> None:
        self.n_dim = n_dim
        self.complex = complex

        if args is None:
            self.args = symbols(f"x:{n_dim}")
        else:
            self.args = args

        # convert to expressions from poly
        if type(constraint_equations) is not list:
            self.codim = 1
            constraint_equations = [constraint_equations.expr]
        else:
            self.codim = len(constraint_equations)
            constraint_equations = [eq.expr for eq in constraint_equations]

        self.constraint_equation = sympy_func_to_array_func(
            self.args, sympy.Matrix(constraint_equations)
        )
        self.jitted_constraint_equation = sympy_func_to_jit_func(
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

    def jacobian(self, p: np.array):
        jacobian_func = getattr(self, "_jacobian_func", None)
        if jacobian_func is None:
            jacobian = symbolic_jacobian(self.algebraic_equation, self.args)
            self._jacobian_func = sympy_func_to_array_func(self.args, jacobian)
            return self._jacobian_func(p)
        else:
            return jacobian_func(p)

    def jitted_jacobian(self, p: np.array):
        jacobian_func = getattr(self, "_jitted_jacobian_func", None)
        if jacobian_func is None:
            jacobian = symbolic_jacobian(self.algebraic_equation, self.args)
            self._jitted_jacobian_func = sympy_func_to_jit_func(self.args, jacobian)
            return self._jitted_jacobian_func(p)
        else:
            return jacobian_func(p)

    def hessian(self, p: np.array):
        hessian_func = getattr(self, "_hessian_func", None)
        if hessian_func is None:
            hessian = symbolic_hessian(self.algebraic_equation, self.args)
            self._hessian_func = sympy_func_to_array_func(self.args, hessian)
            return self._hessian_func(p)
        else:
            return hessian_func(p)

    def shape_operator(self, p: np.array):
        shape_operator_func = getattr(self, "_shape_operator_func", None)
        if shape_operator_func is None:
            shape_operator = symbolic_shape_operator(self.algebraic_equation, self.args)
            self._shape_operator_func = sympy_func_to_array_func(
                self.args, shape_operator
            )
            return self._shape_operator_func(p)
        else:
            return self._shape_operator_func(p)

    def mean_curvature(self, p: np.array):
        mean_curvature_func = getattr(self, "_mean_curvature_func", None)
        if mean_curvature_func is None:
            mean_curvature = symbolic_mean_curvature(self.algebraic_equation, self.args)
            self._mean_curvature_func = sympy_func_to_array_func(
                self.args, mean_curvature
            )
            return self._mean_curvature_func(p)
        else:
            return self._mean_curvature_func(p)

    def gaussian_curvature(self, p: np.array):
        gaussian_curvature_func = getattr(self, "_gaussian_curvature_func", None)
        if gaussian_curvature_func is None:
            gaussian_curvature = symbolic_gaussian_curvature(
                self.algebraic_equation, self.args
            )
            self._gaussian_curvature_func = sympy_func_to_array_func(
                self.args, gaussian_curvature
            )
            return self._gaussian_curvature_func(p)
        else:
            return self._gaussian_curvature_func(p)


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
        constraint_equation = sympy.Poly(x0**power + x1**power + x2**power - 1)
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


class Ellipsoid(AlgebraicSurface):
    def __init__(
        self,
        x_factor: float,
        y_factor: float,
        z_factor: float,
        metric: Optional[Callable] = None,
        tol=DEFAULT_TOLERANCE,
    ) -> None:
        """Stretched sphere shape."""
        constraint_equation = sympy.Poly(
            x_factor * x0**2 + y_factor * x1**2 + z_factor * x2**2 - 1
        )
        super().__init__(
            n_dim=3, constraint_equations=constraint_equation, metric=metric, tol=tol
        )


class SimpleAlgebraicIntersection(AlgebraicSurface):
    def __init__(
        self, metric: Optional[Callable] = None, tol=DEFAULT_TOLERANCE
    ) -> None:
        eq1 = sympy.Poly(x0**4 + x1**3 + x1 * x1 + x0**2 * x2 - x2 - 1)
        eq2 = sympy.Poly(2 * x1 - x2)
        super().__init__(
            n_dim=3, constraint_equations=[eq1, eq2], metric=metric, tol=tol
        )


class OrthogonalGroup(AlgebraicSurface):
    def __init__(
        self, n: int, metric: Optional[Callable] = None, tol=DEFAULT_TOLERANCE
    ) -> None:

        # Add constraints of the form \sum_k^n x_{ik}^2 = 0 \forall i
        squared_component_equations = []
        for i in range(n):
            row_symbols = sympy.symbols(f"x{i}(:{n})")
            eq = sympy.Poly(sum([s**2 for s in row_symbols]))
            squared_component_equations.append(eq - 1)

        # Add constraints of the form \sum_k^n x_{ik}x_{jk} = 0 \forall i \forall j > i
        cross_component_equations = []
        for i in range(n):
            for j in range(i + 1, n):
                i_symbols = sympy.symbols(f"x{i}(:{n})")
                j_symbols = sympy.symbols(f"x{j}(:{n})")
                eq = sympy.Poly(sum([xi * xj for xi, xj in zip(i_symbols, j_symbols)]))
                cross_component_equations.append(eq)

        constraint_equations = squared_component_equations + cross_component_equations

        super().__init__(
            n**2,
            constraint_equations,
            sympy.symbols(f"x:{n}:{n}"),
            metric,
            tol,
        )


class RandomAlgebraicSurface(AlgebraicSurface):
    def __init__(
        self,
        n_dim: int,
        degree: int,
        codim: int,
        metric: Optional[Callable] = None,
        tol=DEFAULT_TOLERANCE,
    ) -> None:
        symbols = sympy.symbols(f"x:{n_dim}")
        constraint_equations = []
        for i in range(codim):
            coeffs = np.random.random(n_dim)
            eq = sympy.Poly(sum([c * x for c, x in zip(coeffs, symbols)]))
            constraint_equations.append(eq)
        super().__init__(n_dim, constraint_equations, metric, tol)

class TrottCurve(AlgebraicSurface):
    def __init__(self, metric: Optional[Callable] = None, tol=DEFAULT_TOLERANCE):
        eq = sympy.Poly(144 * (x0 ** 4 + x1 ** 4) - 225 * (x0 ** 2+ x1 ** 2)+ 350 * x0 ** 2 * x1 ** 2 + 81)
        super().__init__(n_dim=2, constraint_equations=[eq], metric=metric, tol=tol)

class ComplexVariety(AlgebraicSurface):
    def __init__(self, variety: cyjax.VarietySingle, tol=DEFAULT_TOLERANCE) -> None:
        poly = variety.defining_poly.affine_poly(0).sympy_poly()
        re, im = poly.as_expr().as_real_imag()
        re_im_symbols = sympy.symbols(f"x1:{len(poly.args)} y1:{len(poly.args)}")
        re, im = sympy.poly(re), sympy.poly(im)
        re, im = re.subs(zip(re.args[1:], re_im_symbols)), im.subs(zip(im.args[1:], re_im_symbols))
        super().__init__(n_dim=2 * (len(poly.args) - 1), constraint_equations=[re, im], args=re_im_symbols, tol=tol)

_eq1 = x0 ** 4 - x1**2 * x2**2 + x2**3 * x3 + x0 * x2 ** 3 - x1 * x3 + x3 ** 2 - x2 + 1
_eq2 = x3 ** 4 - x1**3 * x2 + x2**3 * x3 + x0 ** 3 - x1 * x0 - x2 + 1
quartic1 = AlgebraicSurface(n_dim=4, constraint_equations=[sympy.Poly(_eq1), sympy.Poly(_eq2)])

_eq3 = -x0 ** 2 * x1 ** 2 - x3 ** 3 * x2 + x0 ** 2 + x2 ** 2 - x1 ** 2 + x3 - 1
_eq4 = x0 * x3 ** 3 - x1 * x2 * x3 + - x0 - x2 + 1
quartic2 = AlgebraicSurface(n_dim=4, constraint_equations=[sympy.Poly(_eq3), sympy.Poly(_eq4)])

quartic_initial_point = np.array([0, 0, 1, 0])
quartic_constraints = [lambda x : 2 + 1e-10 - np.abs(x).max()]
