from geometric_sampling.manifold_sampling.surfaces import AlgebraicSurface
from geometric_sampling.manifold_sampling.utils import (
    sympy_func_to_array_func,
    jitted_norm,
)

import sympy
import numpy as np
from numba import njit, float64
from scipy import optimize

t = sympy.symbols("t")
NEWTON_MAX_ITER = 50


def generate_line_equation_coefficients(surface: AlgebraicSurface):
    """Symbolically generate the coefficients of the polynomial corresponding to the restriction
    of the surface equation to the line p-->q."""
    p, q = sympy.symbols(f"p:{surface.n_dim}"), sympy.symbols(f"q:{surface.n_dim}")
    line = sympy.Matrix(p) + t * (sympy.Matrix(q) - sympy.Matrix(p))
    exp = surface.algebraic_equation[0].subs(
        zip(sympy.symbols(f"x:{surface.n_dim}"), line)
    )
    coeffs = sympy.Poly(exp, t).all_coeffs()
    line_eq_coeffs = sympy_func_to_array_func(p + q, coeffs)
    return line_eq_coeffs


def find_line_intersections(
    line_eq_coeffs: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    return_all: bool = False,
):
    """Find solution to constraint on line p1-->p2, or return None if not found.

    Args:
        p1 (torch.Tensor): first point on sphere
        p2 (torch.Tensor): second point on sphere
    """
    # Find roots
    roots = np.roots(line_eq_coeffs)

    # Only accept roots in sphere
    sols_mask = (
        (np.abs(np.imag(roots)) < 1e-14) & (np.real(roots) >= 0) & (np.real(roots) <= 1)
    )
    sols = np.real(roots[sols_mask])

    if len(sols) == 0:
        return None
    elif return_all:
        return p1 + sols[:, None] * (p2 - p1)
    else:
        return p1 + np.random.choice(sols) * (p2 - p1)


@njit(float64[:](float64[:], float64[:, :], float64[:]))
def _newton_inner_loop(x, j_x, f_val):
    delta = np.linalg.solve(j_x, -f_val)
    x += delta
    return x


def newton_solver(F, J, x, eps):
    """
    Solve nonlinear system F=0 by Newton's method.
    J is the Jacobian of F. Both F and J must be functions of x.
    At input, x holds the start value. The iteration continues
    until ||F|| < eps.
    """
    x = x.copy()
    F_value = F(x)
    F_norm = jitted_norm(F_value)  # l2 norm of vector
    iteration_counter = 0
    while F_norm > eps and iteration_counter < NEWTON_MAX_ITER:
        x = _newton_inner_loop(x, J(x), F_value)
        F_value = F(x)
        F_norm = jitted_norm(F_value)
        iteration_counter += 1

    # Here, either a solution is found, or too many iterations
    if F_norm > eps:
        x = np.NaN
    return x, iteration_counter, iteration_counter


def scipy_solver(F, J, x, eps, method="hybr"):
    result = optimize.root(
        F,
        x,
        method=method,
        options=dict(
            maxfev=NEWTON_MAX_ITER,
            maxiter=NEWTON_MAX_ITER,
        ),
        jac=J,
    )

    if J is None:
        result.njev = 0

    if result.success:
        return result.x, result.nfev, result.njev
    else:
        return np.NaN, result.nfev, result.njev
