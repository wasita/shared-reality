"""
Hierarchical Bayesian Factor Model for Commonality Inference.

Partner's beliefs are encoded as position θ ∈ Rᵏ in factor space.
Factor loadings Λ define how latent factors map to survey responses.
Transfer structure emerges from factor geometry—no manual specification needed.

INFERENCE (Closed-Form Gaussian)
================================
Prior:      θ ~ N(λ·θ_self, σ²_prior·I)
Likelihood: r_obs | θ ~ N(Λ_obs'θ + μ_q, σ²_obs)
Posterior:  θ | r_obs ~ N(μ_post, Σ_post)

When infer_lambda=True, we jointly infer (λ, θ) by marginalizing over a λ grid.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.stats import norm as jax_norm

jax.config.update("jax_platform_name", "cpu")

# ============================================================================
# PATHS AND CONSTANTS
# ============================================================================

PAPER_DIR = Path(__file__).parent.parent
DATA_DIR = PAPER_DIR / "data"
N_QUESTIONS = 35

DOMAIN_RANGES = {
    'arbitrary': (0, 5),
    'background': (5, 10),
    'identity': (10, 15),
    'morality': (15, 20),
    'politics': (20, 25),
    'preferences': (25, 30),
    'religion': (30, 35),
}


# ============================================================================
# JAX-ACCELERATED CORE FUNCTIONS
# ============================================================================

@jit
def _compute_posterior(L_obs, r_centered, mu_0, Sigma_0_inv, sigma_obs_sq):
    """Compute posterior mean and covariance via Gaussian conjugacy."""
    precision_obs = jnp.outer(L_obs, L_obs) / sigma_obs_sq
    Sigma_post_inv = Sigma_0_inv + precision_obs
    Sigma_post = jnp.linalg.inv(Sigma_post_inv)
    mu_post = Sigma_post @ (Sigma_0_inv @ mu_0 + L_obs * r_centered / sigma_obs_sq)
    return mu_post, Sigma_post


@jit
def _predict_all_questions(L, mu_q, mu_post, Sigma_post, r_self_all, tau, sigma_obs_sq):
    """Predict match probabilities for all questions given posterior."""
    pred_means = L @ mu_post + mu_q
    pred_vars = jnp.sum((L @ Sigma_post) * L, axis=1) + sigma_obs_sq
    pred_stds = jnp.sqrt(pred_vars)
    upper = (r_self_all + tau - pred_means) / pred_stds
    lower = (r_self_all - tau - pred_means) / pred_stds
    return jnp.clip(jax_norm.cdf(upper) - jax_norm.cdf(lower), 0.0, 1.0)


@jit
def _marginal_likelihood(L_obs, r_obs, mu_q_obs, theta_self, lam, Sigma_0, sigma_obs_sq):
    """Compute P(r_obs | λ) for lambda posterior calculation."""
    mu_0 = lam * theta_self
    marginal_mean = L_obs @ mu_0 + mu_q_obs
    marginal_var = L_obs @ Sigma_0 @ L_obs + sigma_obs_sq
    return jax_norm.pdf(r_obs, marginal_mean, jnp.sqrt(marginal_var))


def _predict_with_lambda_inference(
    obs_q, r_partner_obs, r_self_all, L, mu_q,
    Sigma_0, Sigma_0_inv, sigma_obs, tau,
    lambda_grid, lambda_prior_probs
):
    """Full prediction pipeline with joint (λ, θ) inference."""
    sigma_obs_sq = sigma_obs ** 2
    L_obs = L[obs_q]
    r_centered = r_partner_obs - mu_q[obs_q]

    # Project self onto factor space
    r_self_centered = r_self_all - mu_q
    LtL_inv = jnp.linalg.inv(L.T @ L + 1e-6 * jnp.eye(L.shape[1]))
    theta_self = LtL_inv @ L.T @ r_self_centered

    # Compute P(λ | r_obs) via marginal likelihoods
    marginal_liks = vmap(
        lambda lam: _marginal_likelihood(L_obs, r_partner_obs, mu_q[obs_q], theta_self, lam, Sigma_0, sigma_obs_sq)
    )(lambda_grid)
    lambda_posterior = marginal_liks * lambda_prior_probs
    lambda_posterior = lambda_posterior / (lambda_posterior.sum() + 1e-10)

    # Predict for each λ, then marginalize
    def predict_for_lambda(lam):
        mu_0 = lam * theta_self
        mu_post, Sigma_post = _compute_posterior(L_obs, r_centered, mu_0, Sigma_0_inv, sigma_obs_sq)
        return _predict_all_questions(L, mu_q, mu_post, Sigma_post, r_self_all, tau, sigma_obs_sq)

    predictions_per_lambda = vmap(predict_for_lambda)(lambda_grid)
    return lambda_posterior @ predictions_per_lambda


# ============================================================================
# DATA LOADING
# ============================================================================

def load_responses() -> pd.DataFrame:
    """Load pre-interaction responses (participants × 35 questions)."""
    df = pd.read_csv(DATA_DIR / "responses.csv", low_memory=False)
    responses = df.pivot_table(index='pid', columns='question', values='preChatResponse', aggfunc='first')
    responses.columns = [f'Q{c}' for c in responses.columns]
    return responses


def load_correlation_matrix() -> np.ndarray:
    """Compute empirical 35×35 correlation matrix from responses."""
    return np.corrcoef(load_responses().values.T)


def load_factor_loadings(k: Optional[int] = None) -> np.ndarray:
    """Compute factor loadings via eigendecomposition of correlation matrix."""
    corr_matrix = load_correlation_matrix()
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    loadings = eigenvectors[:, idx] * np.sqrt(np.maximum(eigenvalues[idx], 0))
    return loadings[:, :k] if k is not None else loadings


def load_question_means() -> np.ndarray:
    """Compute mean response for each question from population."""
    return load_responses().values.mean(axis=0)


def load_unified_data() -> pd.DataFrame:
    """Load unified evaluation dataset with column aliases."""
    df = pd.read_csv(DATA_DIR / "responses.csv", low_memory=False)
    df['own_response'] = df['preChatResponse']
    df['question_domain'] = df['preChatDomain']
    df['matched_question'] = df['matchedIdx']
    return df


# ============================================================================
# MODEL
# ============================================================================

@dataclass
class Posterior:
    """Posterior distribution over partner's factor position."""
    mean: np.ndarray
    covariance: np.ndarray


class BayesianFactorModel:
    """
    Bayesian factor model for commonality inference.

    Parameters
    ----------
    k : int
        Factor dimensionality. k=0 means flat (no structure).
    infer_lambda : bool
        If True, jointly infer self-projection weight λ from data.
        If False, use fixed λ value.
    lam : float
        Fixed λ value when infer_lambda=False.
    sigma_obs : float
        Observation noise σ_obs.
    sigma_prior : float
        Prior standard deviation on factor position.
    match_threshold : float
        τ, threshold for defining a "match".
    epsilon : float
        Response noise / lapse rate ε ∈ [0, 1].
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
        self.sigma_obs = sigma_obs
        self.sigma_prior = sigma_prior
        self.tau = match_threshold
        self.epsilon = np.clip(epsilon, 0.0, 1.0)
        self.infer_lambda = infer_lambda
        self.fixed_lam = np.clip(lam, 0.0, 1.0)

        # Load or use provided data
        self.mu_q = question_means if question_means is not None else load_question_means()

        # Handle factor structure
        if k == 0:
            self.L = np.ones((N_QUESTIONS, 1))
            self.k_effective = 1
        else:
            if loadings is not None:
                self.L = loadings[:, :k] if loadings.shape[1] > k else loadings
            else:
                self.L = load_factor_loadings(k=k)
            self.k_effective = self.L.shape[1]

        # Prior covariance
        self.Sigma_0 = sigma_prior**2 * np.eye(self.k_effective)
        self.Sigma_0_inv = np.eye(self.k_effective) / sigma_prior**2

        # Lambda grid for inference
        self.lambda_grid = np.linspace(0, 1, lambda_grid_size)
        self.lambda_prior_probs = np.ones(lambda_grid_size) / lambda_grid_size

        # Cache JAX arrays
        self._L = jnp.array(self.L)
        self._mu_q = jnp.array(self.mu_q)
        self._Sigma_0 = jnp.array(self.Sigma_0)
        self._Sigma_0_inv = jnp.array(self.Sigma_0_inv)
        self._lambda_grid = jnp.array(self.lambda_grid)
        self._lambda_prior_probs = jnp.array(self.lambda_prior_probs)

    def predict(self, obs_q: int, r_partner_obs: float, r_self_all: np.ndarray) -> np.ndarray:
        """
        Predict match probabilities for all questions.

        Args:
            obs_q: Index of observed question (0-indexed)
            r_partner_obs: Partner's observed response
            r_self_all: Self's responses on all 35 questions

        Returns:
            Array of P(match) for each question
        """
        r_self_jax = jnp.array(r_self_all)

        if self.infer_lambda:
            preds = _predict_with_lambda_inference(
                obs_q, r_partner_obs, r_self_jax,
                self._L, self._mu_q, self._Sigma_0, self._Sigma_0_inv,
                self.sigma_obs, self.tau, self._lambda_grid, self._lambda_prior_probs
            )
        else:
            # Fixed lambda: single posterior update
            L_obs = self._L[obs_q]
            r_centered = r_partner_obs - self._mu_q[obs_q]

            # Compute theta_self for prior mean
            r_self_centered = r_self_jax - self._mu_q
            LtL_inv = jnp.linalg.inv(self._L.T @ self._L + 1e-6 * jnp.eye(self.k_effective))
            theta_self = LtL_inv @ self._L.T @ r_self_centered
            mu_0 = self.fixed_lam * theta_self

            mu_post, Sigma_post = _compute_posterior(
                L_obs, r_centered, mu_0, self._Sigma_0_inv, self.sigma_obs**2
            )
            preds = _predict_all_questions(
                self._L, self._mu_q, mu_post, Sigma_post,
                r_self_jax, self.tau, self.sigma_obs**2
            )

        # Apply lapse rate
        preds = (1 - self.epsilon) * preds + self.epsilon * 0.5
        return np.asarray(preds)

    # Alias for backward compatibility
    def predict_participant(self, obs_q: int, r_partner_obs: float, r_self_all: np.ndarray) -> np.ndarray:
        return self.predict(obs_q, r_partner_obs, r_self_all)

    def __repr__(self):
        if self.infer_lambda:
            return f"BayesianFactorModel(k={self.k}, infer_λ=True, ε={self.epsilon})"
        return f"BayesianFactorModel(k={self.k}, λ={self.fixed_lam}, ε={self.epsilon})"


# ============================================================================
# MODEL CONSTRUCTORS (for ablations)
# ============================================================================

def FullModel(k: int = 4, **kwargs) -> BayesianFactorModel:
    """Full model with joint inference over (λ, θ)."""
    return BayesianFactorModel(k=k, infer_lambda=True, **kwargs)


def FactorModel(k: int = 4, **kwargs) -> BayesianFactorModel:
    """Factor model without self-projection (λ=0)."""
    return BayesianFactorModel(k=k, infer_lambda=False, lam=0.0, **kwargs)


def PopulationBaseline(**kwargs) -> BayesianFactorModel:
    """Population baseline: no structure, no self-projection."""
    return BayesianFactorModel(k=0, infer_lambda=False, lam=0.0, **kwargs)


def EgocentricBaseline(**kwargs) -> BayesianFactorModel:
    """Egocentric baseline: no structure, full self-projection."""
    return BayesianFactorModel(k=0, infer_lambda=False, lam=1.0, **kwargs)


# ============================================================================
# EVALUATION HELPERS
# ============================================================================

def run_evaluation(model, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Run model on all participants and return predictions DataFrame."""
    if data is None:
        data = load_unified_data()

    predictions = []
    for pid in data["pid"].unique():
        subj = data[data["pid"] == pid]
        matched = subj[subj["is_matched"] == True]

        if len(matched) == 0:
            continue

        obs_q = int(matched["matched_question"].iloc[0]) - 1
        partner_obs = matched["partner_response"].iloc[0]

        if pd.isna(partner_obs):
            continue

        match_type = matched["match_type"].iloc[0]

        # Get self's responses
        r_self_all = np.zeros(N_QUESTIONS)
        for _, row in subj.iterrows():
            r_self_all[int(row["question"]) - 1] = row["own_response"]

        pred_probs = model.predict(obs_q, float(partner_obs), r_self_all)

        for _, row in subj.iterrows():
            q_idx = int(row["question"]) - 1
            predictions.append({
                "pid": pid,
                "question": row["question"],
                "question_domain": row["question_domain"],
                "match_type": match_type,
                "question_type": row["question_type"],
                "pred_prob": pred_probs[q_idx],
                "actual": row["participant_binary_prediction"],
            })

    return pd.DataFrame(predictions)


def compute_metrics(pred_df: pd.DataFrame, human_rates: Optional[Dict] = None) -> Dict:
    """Compute evaluation metrics from predictions DataFrame."""
    results = {}

    probs = pred_df["pred_prob"].values
    actual = pred_df["actual"].values

    # Trial-level metrics
    eps = 1e-10
    ll = np.sum(actual * np.log(probs + eps) + (1 - actual) * np.log(1 - probs + eps))
    results['log_likelihood'] = ll
    results['accuracy'] = np.mean((probs > 0.5) == actual)
    results['brier'] = np.mean((probs - actual) ** 2)

    # Cell-level rates
    model_rates = {}
    for qt in ['observed', 'same_domain', 'different_domain']:
        for mt in ['high', 'low']:
            cell = pred_df[(pred_df["question_type"] == qt) & (pred_df["match_type"] == mt)]
            model_rates[(qt, mt)] = cell["pred_prob"].mean() if len(cell) > 0 else 0.5

    results['model_rates'] = model_rates

    # Effects
    for qt in ['observed', 'same_domain', 'different_domain']:
        results[f'{qt}_effect'] = model_rates[(qt, 'high')] - model_rates[(qt, 'low')]

    # Correlation with human rates if provided
    if human_rates:
        m = [model_rates[k] for k in sorted(human_rates.keys())]
        h = [human_rates[k] for k in sorted(human_rates.keys())]
        results['correlation'] = float(np.corrcoef(m, h)[0, 1])

    return results
