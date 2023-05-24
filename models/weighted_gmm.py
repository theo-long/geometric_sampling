import numpy as np
import warnings

from geometric_sampling.manifold_sampling.solve import generate_line_equation_coefficients
from geometric_sampling.manifold_sampling.surfaces import ConstraintSurface
from geometric_sampling.manifold_sampling.sampler import sphere_sample

from scipy.stats.sampling import NumericalInversePolynomial

from sklearn.mixture import GaussianMixture
from sklearn.mixture._base import check_random_state
from sklearn.mixture._gaussian_mixture import (
    _estimate_gaussian_covariances_full, 
    _estimate_gaussian_covariances_tied, 
    _estimate_gaussian_covariances_diag,
    _compute_precision_cholesky,
)

class RDist:
    def __init__(self, dim, variance):
        self.dim = dim
        self.variance = variance
        self.sampler = NumericalInversePolynomial(self)

    def pdf(self, r):
        if r < 0:
            return 0
        return r**(self.dim + 1) * np.exp( - r ** 2 / (2 * self.variance))
    
    def sample(self, size):
        return self.sampler.rvs(size=size)

def _estimate_gaussian_parameters(X, sample_weights, resp, reg_covar, covariance_type):
    nk = (resp * sample_weights[:, None]).sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    means = np.dot(resp.T, X * sample_weights[:, None]) / nk[:, np.newaxis]
    covariances = {
        "full": _estimate_gaussian_covariances_full,
        "tied": _estimate_gaussian_covariances_tied,
        "diag": _estimate_gaussian_covariances_diag,
        "spherical": _estimate_gaussian_covariances_spherical,
    }[covariance_type](resp, X, sample_weights, nk, means, reg_covar)
    return nk, means, covariances

def _estimate_gaussian_covariances_spherical(resp, X, sample_weights, nk, means, reg_covar):
    avg_X2 = np.dot(resp.T, X * X * sample_weights[:, None]) / nk[:, np.newaxis]
    avg_means2 = means**2
    avg_X_means = means * np.dot(resp.T, X * sample_weights[:, None]) / nk[:, np.newaxis]
    diag_cov = avg_X2 - 2 * avg_X_means + avg_means2 + reg_covar
    return diag_cov.mean(1)

class WeightedGMM(GaussianMixture):
    
    def sample(self, n_samples=1):
        if n_samples < 1:
            raise ValueError(
                "Invalid value for 'n_samples': %d . The sampling requires at "
                "least one sample." % (self.n_components)
            )

        _, n_features = self.means_.shape
        rng = check_random_state(self.random_state)
        n_samples_comp = rng.multinomial(n_samples, self.weights_)

        if self.covariance_type == "full":
            X = np.vstack(
                [
                    rng.multivariate_normal(mean, covariance, int(sample))
                    for (mean, covariance, sample) in zip(
                        self.means_, self.covariances_, n_samples_comp
                    )
                ]
            )
        elif self.covariance_type == "tied":
            X = np.vstack(
                [
                    rng.multivariate_normal(mean, self.covariances_, int(sample))
                    for (mean, sample) in zip(self.means_, n_samples_comp)
                ]
            )
        else:
            X = np.vstack(
                [
                    mean
                    + rng.standard_normal(size=(sample, n_features))
                    * np.sqrt(covariance)
                    for (mean, covariance, sample) in zip(
                        self.means_, self.covariances_, n_samples_comp
                    )
                ]
            )

        y = np.concatenate(
            [np.full(sample, j, dtype=int) for j, sample in enumerate(n_samples_comp)]
        )

        return (X, y)
    
    def fit(self, X, sample_weights, y=None):
        self.fit_predict(X, sample_weights, y)
        return self

    def fit_predict(self, X, sample_weights, y=None):
        X = self._validate_data(X, dtype=[np.float64, np.float32], ensure_min_samples=2)
        if X.shape[0] < self.n_components:
            raise ValueError(
                "Expected n_samples >= n_components "
                f"but got n_components = {self.n_components}, "
                f"n_samples = {X.shape[0]}"
            )
        self._check_initial_parameters(X)

        # if we enable warm_start, we will have a unique initialisation
        do_init = not (self.warm_start and hasattr(self, "converged_"))
        n_init = self.n_init if do_init else 1

        max_lower_bound = -np.inf
        self.converged_ = False

        random_state = check_random_state(self.random_state)

        n_samples, _ = X.shape
        for init in range(n_init):
            self._print_verbose_msg_init_beg(init)

            if do_init:
                self._initialize_parameters(X, random_state)

            lower_bound = -np.inf if do_init else self.lower_bound_

            if self.max_iter == 0:
                best_params = self._get_parameters()
                best_n_iter = 0
            else:
                for n_iter in range(1, self.max_iter + 1):
                    prev_lower_bound = lower_bound

                    log_prob_norm, log_resp = self._e_step(X)
                    self._m_step(X, sample_weights, log_resp)
                    lower_bound = self._compute_lower_bound(log_resp, log_prob_norm)

                    change = lower_bound - prev_lower_bound
                    self._print_verbose_msg_iter_end(n_iter, change)

                    if abs(change) < self.tol:
                        self.converged_ = True
                        break

                self._print_verbose_msg_init_end(lower_bound)

                if lower_bound > max_lower_bound or max_lower_bound == -np.inf:
                    max_lower_bound = lower_bound
                    best_params = self._get_parameters()
                    best_n_iter = n_iter

        # Should only warn about convergence if max_iter > 0, otherwise
        # the user is assumed to have used 0-iters initialization
        # to get the initial means.
        if not self.converged_ and self.max_iter > 0:
            warnings.warn(
                "Initialization %d did not converge. "
                "Try different init parameters, "
                "or increase max_iter, tol "
                "or check for degenerate data." % (init + 1)
            )

        self._set_parameters(best_params)
        self.n_iter_ = best_n_iter
        self.lower_bound_ = max_lower_bound

        # Always do a final e-step to guarantee that the labels returned by
        # fit_predict(X) are always consistent with fit(X).predict(X)
        # for any value of max_iter and tol (and any random_state).
        _, log_resp = self._e_step(X)

        return log_resp.argmax(axis=1)
   
    def _m_step(self, X, sample_weights, log_resp):
        """M step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        log_resp : array-like of shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        self.weights_, self.means_, self.covariances_ = _estimate_gaussian_parameters(
            X, sample_weights, np.exp(log_resp), self.reg_covar, self.covariance_type
        )
        self.weights_ /= self.weights_.sum()
        
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type
        )
        
    def surface_sample(self, surface: ConstraintSurface, n_samples=1):
        if n_samples < 1:
            raise ValueError(
                "Invalid value for 'n_samples': %d . The sampling requires at "
                "least one sample." % (self.n_components)
            )

        _, n_features = self.means_.shape
        rng = check_random_state(self.random_state)
        n_samples_comp = rng.multinomial(n_samples, self.weights_)
    
        # Set up constraint solver
        line_eq_coeffs = generate_line_equation_coefficients(surface)    
        
        points = []
        for mean, covariance, sample in zip(self.means_, self.covariances_, n_samples_comp):
            radius_sampler = RDist(dim=surface.n_dim - 1, variance=covariance)
            p, _ = sphere_sample(
                n_samples=n_samples, 
                surface=surface,
                centre=mean,
                radius = radius_sampler.sample, 
                line_eq_coeffs=line_eq_coeffs, 
            )
            points.append(p)
            
        return np.concatenate(points)
    
    def log_pdf(self, X):
        return self.score_samples(X)
    
    def pdf(self, X):
        return np.exp(self.log_pdf(X))