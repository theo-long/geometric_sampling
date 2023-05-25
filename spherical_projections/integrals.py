import numpy as np
import torch

from geometric_sampling.manifold_sampling.surfaces import AlgebraicSurface
from geometric_sampling.spherical_projections.geometry import (
    theta_to_phi,
    sqrt_detJ_func,
    dphi_dtheta_func,
)


def generate_coarea_factor(
    theta_val, surface: AlgebraicSurface, surface_point: np.ndarray
):
    """This is the scaling factor given by the Riemannian Coarea Formula."""

    tangent_space, normal_space, n_hat = surface.generate_tangent_and_normal_space(
        surface_point
    )

    h_val = -np.dot(surface_point, n_hat.squeeze())

    phi_val = theta_to_phi(theta_val, surface_point[0], surface_point[1])

    angle_delta = np.arccos(-tangent_space[0][0])
    theta_prime = theta_val + angle_delta
    phi_prime = phi_val + angle_delta

    det_jac = sqrt_detJ_func(h_val, phi_prime, theta_prime)

    dphi_dtheta_val = dphi_dtheta_func(theta_val, surface_point[0], surface_point[1])

    return np.sqrt(dphi_dtheta_val**2 + 1) / det_jac, phi_val


def integrand(theta_val, surface: AlgebraicSurface, surface_point, density_func=None):
    coarea_factor, phi_val = generate_coarea_factor(theta_val, surface, surface_point)
    if density_func:
        prob_theta_phi = density_func(theta_val, phi_val)
    else:
        prob_theta_phi = 1
    return prob_theta_phi * coarea_factor


def calculate_loss(
    phi_model: torch.nn.Module,
    theta_model: torch.nn.Module,
    phi_vals: torch.Tensor,
    theta_vals: torch.Tensor,
    coarea_factors: torch.Tensor,
    phi_indices: torch.Tensor,
    target_surface_probabilities: torch.Tensor,
    final_activation=torch.sigmoid,
    device=None,
    pass_as_xy=False,
):

    n_surface_points = coarea_factors.shape[0]

    # Calculate probabilities output by models for each pair theta_val, phi_val
    # Note - we reshape both to an N_surface x N_theta matrix for calculating loss
    if pass_as_xy:
        phi_probs = final_activation(
            torch.concatenate([torch.cos(phi_vals), torch.sin(phi_vals)], dim=-1)
        ).squeeze()
        theta_probs = final_activation(
            torch.concatenate([torch.cos(theta_vals), torch.sin(theta_vals)], dim=-1)
        ).squeeze()
    else:
        phi_probs = final_activation(phi_model(phi_vals)).squeeze()
        theta_probs = final_activation(
            theta_model(theta_vals)
        ).squeeze()

    phi_probs = phi_probs / phi_probs.sum()
    theta_probs = theta_probs / theta_probs.sum()

    # Loss function
    # 1. we calculate the model-based probability at each point on the surface
    # 2. Take KL divergence of this and the true distribution
    theta_probabilities = coarea_factors * theta_probs
    calculated_surface_probabilities = torch.zeros(n_surface_points).to(device)

    # for loop to save memory
    for i in range(n_surface_points):
        row_phi_indices = phi_indices[i]
        calculated_surface_probabilities[i] = (
            theta_probabilities[i] * phi_probs[row_phi_indices]
        ).mean()

    # Normalize probabilities
    calculated_surface_probabilities /= calculated_surface_probabilities.sum()

    # KL divergence loss
    loss = (
        target_surface_probabilities
        * torch.log(target_surface_probabilities / calculated_surface_probabilities)
    ).sum()

    return loss
