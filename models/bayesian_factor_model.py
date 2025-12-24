"""
Bayesian Factor Model for Commonality Inference.

Given a single observation of a partner's response, predict how likely they are
to match self on all other questions. Transfer structure emerges from the
geometry of factor loadings—questions that load on similar factors show
correlated predictions.

MODEL
=====
Partner's latent position:  θ ∈ Rᵏ
Factor loadings:            Λ ∈ R^{35 × k}
Question means:             μ ∈ R^35

Prior:      θ ~ N(λ·θ_self, σ²_prior·I)
Likelihood: r | θ ~ N(Λθ + μ, σ²_obs·I)
Posterior:  θ | r_obs ∝ Likelihood × Prior  (closed-form Gaussian)

The self-projection weight λ ∈ [0,1] controls how much the prior centers on
self's position. When infer_lambda=True, we marginalize over λ using Bayes rule.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.stats import norm as jax_norm

jax.config.update("jax_platform_name", "cpu")


# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
N_QUESTIONS = 35

# Domain structure (for evaluation)
DOMAIN_RANGES = {
    'arbitrary': (0, 5), 'background': (5, 10), 'identity': (10, 15),
    'morality': (15, 20), 'politics': (20, 25), 'preferences': (25, 30),
    'religion': (30, 35),
}


# =============================================================================
# INFERENCE (JAX-accelerated)
# =============================================================================

@jit
def _project_to_factors(responses, loadings, means):
    """Project responses onto factor space via least-squares: θ = (Λ'Λ)⁻¹Λ'(r - μ)"""
    centered = responses - means
    LtL_inv = jnp.linalg.inv(loadings.T @ loadings + 1e-6 * jnp.eye(loadings.shape[1]))
    return LtL_inv @ loadings.T @ centered


@jit
def _posterior_update(L_obs, r_obs, mu_obs, prior_mean, prior_precision, obs_variance):
    """
    Bayesian update: P(θ | r_obs) ∝ P(r_obs | θ) P(θ)

    Returns posterior mean and covariance.
    """
    r_centered = r_obs - mu_obs
    obs_precision = jnp.outer(L_obs, L_obs) / obs_variance
    post_precision = prior_precision + obs_precision
    post_cov = jnp.linalg.inv(post_precision)
    post_mean = post_cov @ (prior_precision @ prior_mean + L_obs * r_centered / obs_variance)
    return post_mean, post_cov


@jit
def _predict_match_probs(loadings, means, post_mean, post_cov, r_self, threshold, obs_variance):
    """
    Predict P(|r_partner - r_self| ≤ τ) for each question.

    Predictive distribution: r_q ~ N(Λ_q'θ + μ_q, Λ_q'Σ_post Λ_q + σ²)
    """
    pred_means = loadings @ post_mean + means
    pred_vars = jnp.sum((loadings @ post_cov) * loadings, axis=1) + obs_variance
    pred_stds = jnp.sqrt(pred_vars)

    upper = (r_self + threshold - pred_means) / pred_stds
    lower = (r_self - threshold - pred_means) / pred_stds
    return jnp.clip(jax_norm.cdf(upper) - jax_norm.cdf(lower), 0.0, 1.0)


@jit
def _marginal_likelihood_of_lambda(L_obs, r_obs, mu_obs, theta_self, lam, prior_cov, obs_variance):
    """P(r_obs | λ) = ∫ P(r_obs | θ) P(θ | λ) dθ  — marginalizes over θ."""
    prior_mean = lam * theta_self
    pred_mean = L_obs @ prior_mean + mu_obs
    pred_var = L_obs @ prior_cov @ L_obs + obs_variance
    return jax_norm.pdf(r_obs, pred_mean, jnp.sqrt(pred_var))


@jit
def _predict_with_lambda_grid(
    obs_q, r_obs, r_self, loadings, means,
    prior_cov, prior_precision, obs_variance, threshold,
    lambda_grid, lambda_prior
):
    """
    Full prediction: marginalize over λ grid, then over θ.

    1. Compute θ_self (self's position in factor space)
    2. For each λ: compute P(r_obs | λ) and P(predictions | r_obs, λ)
    3. Weight predictions by P(λ | r_obs) ∝ P(r_obs | λ) P(λ)
    """
    L_obs = loadings[obs_q]
    mu_obs = means[obs_q]
    theta_self = _project_to_factors(r_self, loadings, means)

    # Posterior over λ
    marginal_liks = vmap(
        lambda lam: _marginal_likelihood_of_lambda(
            L_obs, r_obs, mu_obs, theta_self, lam, prior_cov, obs_variance
        )
    )(lambda_grid)
    lambda_posterior = marginal_liks * lambda_prior
    lambda_posterior = lambda_posterior / (lambda_posterior.sum() + 1e-10)

    # Predictions for each λ
    def predict_given_lambda(lam):
        prior_mean = lam * theta_self
        post_mean, post_cov = _posterior_update(
            L_obs, r_obs, mu_obs, prior_mean, prior_precision, obs_variance
        )
        return _predict_match_probs(
            loadings, means, post_mean, post_cov, r_self, threshold, obs_variance
        )

    preds_per_lambda = vmap(predict_given_lambda)(lambda_grid)
    return lambda_posterior @ preds_per_lambda


# =============================================================================
# DATA LOADING
# =============================================================================

def load_responses() -> pd.DataFrame:
    """Load response matrix (participants × 35 questions)."""
    df = pd.read_csv(DATA_DIR / "responses.csv", low_memory=False)
    return df.pivot_table(index='pid', columns='question', values='preChatResponse', aggfunc='first')


def load_correlation_matrix() -> np.ndarray:
    """Compute 35×35 correlation matrix from responses."""
    return np.corrcoef(load_responses().values.T)


def load_factor_loadings(k: Optional[int] = None) -> np.ndarray:
    """Compute factor loadings via eigendecomposition: Λ = V·√eigenvalues."""
    eigvals, eigvecs = np.linalg.eigh(load_correlation_matrix())
    idx = np.argsort(eigvals)[::-1]
    loadings = eigvecs[:, idx] * np.sqrt(np.maximum(eigvals[idx], 0))
    return loadings[:, :k] if k else loadings


def load_question_means() -> np.ndarray:
    """Population mean for each question."""
    return load_responses().values.mean(axis=0)


def load_evaluation_data() -> pd.DataFrame:
    """Load data for model evaluation."""
    return pd.read_csv(DATA_DIR / "responses.csv", low_memory=False)


# =============================================================================
# MODEL
# =============================================================================

class BayesianFactorModel:
    """
    Bayesian factor model for commonality inference.

    Parameters
    ----------
    k : int
        Number of factors (0 = flat baseline with no structure)
    infer_lambda : bool
        If True, infer self-projection weight λ from data
    lam : float
        Fixed λ when infer_lambda=False
    sigma_obs, sigma_prior : float
        Observation and prior noise standard deviations
    match_threshold : float
        τ for defining a "match" (|r_partner - r_self| ≤ τ)
    epsilon : float
        Lapse rate (probability of random response)
    """

    def __init__(
        self,
        k: int = 4,
        infer_lambda: bool = True,
        lam: float = 0.0,
        sigma_obs: float = 0.3,
        sigma_prior: float = 2.0,
        match_threshold: float = 1.5,
        epsilon: float = 0.4,
        loadings: Optional[np.ndarray] = None,
        question_means: Optional[np.ndarray] = None,
        lambda_grid_size: int = 21,
    ):
        self.k = k
        self.epsilon = np.clip(epsilon, 0.0, 1.0)

        # Load data
        means = question_means if question_means is not None else load_question_means()
        if k == 0:
            L = np.ones((N_QUESTIONS, 1))  # Flat: all questions identical
        elif loadings is not None:
            L = loadings[:, :k]
        else:
            L = load_factor_loadings(k)

        k_eff = L.shape[1]

        # Set up λ grid (single value if not inferring)
        if infer_lambda:
            lam_grid = np.linspace(0, 1, lambda_grid_size)
            lam_prior = np.ones(lambda_grid_size) / lambda_grid_size
        else:
            lam_grid = np.array([np.clip(lam, 0, 1)])
            lam_prior = np.array([1.0])

        # Cache as JAX arrays
        self._loadings = jnp.array(L)
        self._means = jnp.array(means)
        self._prior_cov = jnp.array(sigma_prior**2 * np.eye(k_eff))
        self._prior_precision = jnp.array(np.eye(k_eff) / sigma_prior**2)
        self._obs_variance = sigma_obs**2
        self._threshold = match_threshold
        self._lambda_grid = jnp.array(lam_grid)
        self._lambda_prior = jnp.array(lam_prior)

        # For repr
        self._infer_lambda = infer_lambda
        self._fixed_lam = lam

    def predict(self, obs_q: int, r_partner: float, r_self: np.ndarray) -> np.ndarray:
        """
        Predict P(match) for all questions given one observation.

        Args:
            obs_q: Which question was observed (0-indexed)
            r_partner: Partner's response on that question
            r_self: Self's responses on all 35 questions

        Returns:
            35-element array of match probabilities
        """
        preds = _predict_with_lambda_grid(
            obs_q, r_partner, jnp.array(r_self),
            self._loadings, self._means,
            self._prior_cov, self._prior_precision, self._obs_variance, self._threshold,
            self._lambda_grid, self._lambda_prior
        )
        # Apply lapse rate
        preds = (1 - self.epsilon) * preds + self.epsilon * 0.5
        return np.asarray(preds)

    def __repr__(self):
        if self._infer_lambda:
            return f"BayesianFactorModel(k={self.k}, infer_λ=True)"
        return f"BayesianFactorModel(k={self.k}, λ={self._fixed_lam})"


