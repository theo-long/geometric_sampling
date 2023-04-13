import torch
import sympy
import numpy as np
import scipy

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
    f = sympy.lambdify(args, expr, modules=["scipy", "numpy"], cse=True)
    def array_f(a):
        return f(*a) # We assume rows correspond to different args
    return array_f

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