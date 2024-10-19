import torch
import sympy
import numpy as np
import scipy
from numba import njit, float64

def grad(f):
    """Wrapper that returns a function computing gradient of f."""
    def result(x):
        x = torch.tensor(x, dtype=torch.float32)
        x_ = x.detach().requires_grad_(True) 
        f(x_).backward()
        return x_.grad.detach()
    return result

def sympy_func_to_array_func(args, expr: sympy.Expr):
    """Turn a sympy expression into an array function.

    Args:
        args: arguments (usually a list of sympy symbols)
        expr (sympy.Expr): sympy expression
    """
    f = sympy.lambdify(args, expr, modules=["scipy", "numpy"])
    def array_f(a):
        return f(*a) # We assume rows correspond to different args
    return array_f

def sympy_func_to_jit_func(args, expr: sympy.Expr):
    """Turn a sympy expression into a jitted numba function.

    Args:
        args: arguments (usually a list of sympy symbols)
        expr (sympy.Expr): sympy expression
    """
    f = njit(sympy.lambdify(args, expr, modules=["scipy", "numpy"]))
    def unpacked_f(a):
        return f(*a) # We assume rows correspond to different args

    try:
        unpacked_f(np.ones(len(args)))
    except AssertionError:
        print("Assertion Fallback")
        # Fallback for a case numba doesn't like

        matrix_str = "np." + repr(np.array(expr)).replace(", dtype=object", "")
        symbol_str = ", ".join([str(s) for s in args])

        func_str = f"""lambda {symbol_str}: {matrix_str}"""
        f = eval(func_str)

        def unpacked_f(a):
            return f(*a)

    return unpacked_f

def torus_surface_element(U, V, r, R):
    """Surface element for integrating 3d torus in U,V plane"""
    return r * (R + np.cos(V))

def torus_wrapper(f, r, R):
    """Wrapper that turns an (x, y, z) torus function to a (u, v) function."""
    def torus_f(U, V):
        X = (R+r*np.cos(V))*np.cos(U)
        Y = (R+r*np.cos(V))*np.sin(U)
        Z = r*np.sin(V)
        return f((X, Y, Z))
    return torus_f

def torus_integral(integral_func, r, R):
    """Integrate function of (x,y,z) coordinates over torus."""
    torus_f = torus_wrapper(integral_func, r=r, R=R)
    val, err = scipy.integrate.dblquad(
        func=lambda u, v: torus_f(u, v) * torus_surface_element(u, v, r, R),
        a=0,
        b=2 * np.pi,
        gfun=0,
        hfun=2*np.pi
    )
    return val, err

@njit
def jitted_norm(x) :
    return np.linalg.norm(x)

def change_affine_coordinates(current_point):
    z_coords = current_point[:current_point.shape[-1]//2] * 1.j + current_point[current_point.shape[-1]//2:]
    z_norms = np.abs(z_coords)
    if z_norms.max() <= 1:
        return current_point

    new_coords = _change_affine_z_coordinates(z_coords, z_norms)
    return np.concatenate([new_coords.imag, new_coords.real], axis=-1)

@njit
def _change_affine_z_coordinates(z_coords, z_norms):
    new_patch = z_norms.argmax()
    new_coords = z_coords / z_coords[new_patch]
    new_coords[new_patch] = 1 / z_coords[new_patch]
    return new_coords