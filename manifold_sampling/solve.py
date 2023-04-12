from geometric_sampling.manifold_sampling.surfaces import AlgebraicSurface
from geometric_sampling.manifold_sampling.utils import sympy_func_to_array_func

import sympy
import numpy as np

t = sympy.symbols('t')

def generate_line_equation_coefficients(surface: AlgebraicSurface):
    """Symbolically generate the coefficients of the polynomial corresponding to the restriction
    of the surface equation to the line p-->q."""
    p, q = sympy.symbols(f"p:{surface.n_dim}"), sympy.symbols(
        f"q:{surface.n_dim}"
    )
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
    return_all: bool=False,
):
    """Find solution to constraint on line p1-->p2, or return None if not found.

    Args:
        p1 (torch.Tensor): first point on sphere
        p2 (torch.Tensor): second point on sphere
    """
    # Find sign changes
    poly = np.polynomial.Polynomial(line_eq_coeffs[::-1])
    roots = poly.roots()

    sols_mask = (np.real_if_close(roots) == np.real(roots)) & (np.real(roots) >= 0) & (np.real(roots) <= 1)
    sols = np.real(roots[sols_mask])

    if len(sols) == 0:
        return None
    elif return_all:
        return p1 + sols[:, None] * (p2 - p1)
    else:
        return p1 + np.random.choice(sols) * (p2 - p1)
