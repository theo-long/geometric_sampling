from geometric_sampling.manifold_sampling.errors import ConstraintError
from typing import Callable, Optional
import torch

class ConstraintSurface:
    
    def __init__(self, n_dim:int, constraint_equation: Callable[[torch.Tensor], torch.Tensor], metric: Optional[Callable]=None, tol=1e-6) -> None:
        self.n_dim = n_dim
        self.constraint_equation = constraint_equation
        self.metric = metric
        self.tol = tol

    def generate_tangent_space(self, x: torch.Tensor):
        '''Generates the tangent space at a point x, given as an orthnormal basis.'''
        normal = torch.tensor(normal)
        Q, R = torch.linalg.qr(normal, mode='complete')
        Q = Q[:,-self.n_dim+1:]
        return Q

    def generate_normal_space(self, x: torch.Tensor):
        '''Generates the subspace normal to the tangent space at x, given as an orthonormal basis.'''
        x = x.requires_grad_()
        constraint_value = self.constraint_equation(x)
        if constraint_value > self.tol:
            raise ConstraintError(f"Point x does not satisfy constraint equation with tolerance {self.tol:.2E}")

        constraint_value.backward()
        result = x.grad
        x = x.requires_grad_(False)
        return result