import torch
import sympy

def grad(f):
    """Wrapper that returns a function computing gradient of f."""
    def result(x):
        x = torch.tensor(x)
        x_ = x.detach().requires_grad_(True) 
        f(x_).backward()
        return x_.grad
    return result

def sympy_func_to_array_func(expr: sympy.Expr):
    """Turn a sympy expression into an array function.

    Args:
        expr (sympy.Expr): _description_
    """
    f = sympy.lambdify(expr.args[1:], expr)
    array_f = lambda a : f(*a) # We assume rows correspond to different args
    return array_f