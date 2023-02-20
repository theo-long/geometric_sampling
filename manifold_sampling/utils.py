import torch

def grad(f):
    """Wrapper that returns a function computing gradient of f."""
    def result(x):
        x = torch.tensor(x)
        x_ = x.detach().requires_grad_(True) 
        f(x_).backward()
        return x_.grad
    return result