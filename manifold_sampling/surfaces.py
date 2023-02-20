from geometric_sampling.manifold_sampling.errors import ConstraintError
from typing import Callable, Optional
import torch

class ConstraintSurface:
    
    def __init__(self, n_dim:int, constraint_equation: Callable[[torch.Tensor], torch.Tensor], metric: Optional[Callable]=None, tol=1e-6) -> None:
        self.n_dim = n_dim
        self.constraint_equation = constraint_equation
        if metric:
            self.metric = metric
        else:
            self.metric =  _euclidean_metric
        self.tol = tol

    def generate_tangent_space(self, x: torch.Tensor):
        '''Generates the tangent space at a point x, given as an orthnormal basis.'''
        normal = self.generate_normal_space(x)
        Q, R = torch.linalg.qr(normal, mode='complete')
        Q = Q[:,-self.n_dim+1:]
        return Q

    def generate_normal_space(self, x: torch.Tensor):
        '''Generates the subspace normal to the tangent space at x, given as an orthonormal basis.'''
        x = x.requires_grad_()
        constraint_value = self._check_constraint(x)
        constraint_value.backward()
        result = x.grad
        x = x.requires_grad_(False)
        return result

    def _check_constraint(self, x):
        constraint_value = self.constraint_equation(x)
        if constraint_value > self.tol:
            raise ConstraintError(f"Point x does not satisfy constraint equation with tolerance {self.tol:.2E}")
        return constraint_value



def _euclidean_metric(x):
    return torch.eye(x.shape[0] - 1)