import torch
import sympy

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
    f = sympy.lambdify(args, expr)
    array_f = lambda a : f(*a) # We assume rows correspond to different args
    return array_f