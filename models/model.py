"""
Commonality Inference Model.

Predicts how observing a partner's response on one question informs beliefs
about their responses on other questions. The model combines two mechanisms:

1. BAYESIAN FACTOR MODEL (β=0): Uses population-level factor structure.
   Transfer emerges from the geometry of factor loadings—questions that load
   on similar factors show correlated predictions. Predicts domain-specific
   generalization gradients based on population covariance.

2. SIMILARITY PROJECTION (β=1): Self-based inference combining:
   - Global perceived similarity from observed agreement (Ames 2004; Tamir & Mitchell 2013)
   - Local self-response similarity for question-specific transfer
   Gradients emerge from the structure of one's OWN beliefs, not population statistics.

The mixture parameter λ ∈ [0,1] blends between these:
    P(match) = (1-λ) × Bayesian + λ × SimilarityProjection

BAYESIAN COMPONENT
==================
Partner's latent position:  θ ∈ Rᵏ
Factor loadings:            Λ ∈ R^{35 × k}
Question means:             μ ∈ R^35

Prior:      θ ~ N(0, σ²_prior·I)
Likelihood: r | θ ~ N(Λθ + μ, σ²_obs·I)
Posterior:  θ | r_obs ∝ Likelihood × Prior  (closed-form Gaussian)

SIMILARITY PROJECTION COMPONENT
===============================
perceived_similarity = exp(-|r_obs - r_self[obs]| / scale)  # Global
self_similarity[q] = exp(-|r_self[q] - r_self[obs]| / scale)  # Local
P(match on q) = base_rate + perceived_similarity × self_similarity[q] × weight

Key: Gradients emerge from self-response structure (within-person correlations).
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
def _posterior_update_delta(L_obs, r_obs, mu_obs, prior_mean, prior_cov):
    """
    Bayesian update with delta function observation (exact, no noise).

    Uses Kalman-style update in the infinite precision limit.
    Appropriate for no-chat condition where observations are explicit.
    """
    r_centered = r_obs - mu_obs

    # Kalman gain: K = Σ_prior @ L / (L' @ Σ_prior @ L)
    L_cov_L = L_obs @ prior_cov @ L_obs  # scalar: L'ΣL
    K = prior_cov @ L_obs / (L_cov_L + 1e-10)  # (k,) vector
    innovation = r_centered - L_obs @ prior_mean
    post_mean = prior_mean + K * innovation

    # Posterior covariance: Σ_post = Σ_prior - K @ L' @ Σ_prior
    post_cov = prior_cov - jnp.outer(K, L_obs @ prior_cov)

    return post_mean, post_cov


@jit
def _posterior_update_gaussian(L_obs, r_obs, mu_obs, prior_mean, prior_cov, obs_variance):
    """
    Standard Bayesian update with Gaussian observation noise.

    Appropriate for chat condition where observations are inferred from conversation.
    """
    r_centered = r_obs - mu_obs
    prior_precision = jnp.linalg.inv(prior_cov)
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
def _predict_bayesian(
    obs_q, r_obs, r_self, loadings, means,
    prior_cov, obs_variance, threshold
):
    """
    Bayesian factor model prediction for a single participant.

    When obs_variance=0: Uses delta function (exact observation, no noise).
        Appropriate for no-chat condition where participants see explicit responses.
    When obs_variance>0: Uses Gaussian observation model.
        Appropriate for chat condition where responses are inferred from conversation.
    """
    L_obs = loadings[obs_q]
    mu_obs = means[obs_q]
    prior_mean = jnp.zeros(loadings.shape[1])

    # Choose update based on obs_variance
    post_mean, post_cov = jax.lax.cond(
        obs_variance < 1e-8,
        lambda _: _posterior_update_delta(L_obs, r_obs, mu_obs, prior_mean, prior_cov),
        lambda _: _posterior_update_gaussian(L_obs, r_obs, mu_obs, prior_mean, prior_cov, obs_variance),
        operand=None
    )

    # Predict match probabilities
    pred_means = loadings @ post_mean + means
    pred_vars = jnp.sum((loadings @ post_cov) * loadings, axis=1) + obs_variance
    pred_vars = jnp.maximum(pred_vars, 1e-10)  # Numerical stability
    pred_stds = jnp.sqrt(pred_vars)

    upper = (r_self + threshold - pred_means) / pred_stds
    lower = (r_self - threshold - pred_means) / pred_stds
    return jnp.clip(jax_norm.cdf(upper) - jax_norm.cdf(lower), 0.0, 1.0)


@jit
def _predict_similarity_projection(obs_q, r_obs, r_self, base_rate, projection_weight, threshold):
    """
    Similarity-modulated projection with self-transfer.

    Combines two self-based mechanisms:
    1. GLOBAL: Perceived similarity from observed agreement (Ames 2004; Tamir & Mitchell 2013)
       "You agreed with me, so you're like me"
    2. LOCAL: Self-response similarity for question-specific transfer
       "Questions I answered similarly will have similar outcomes"

    The model: P(match_q) = base + perceived_similarity × self_similarity[q] × weight

    This produces gradients from the structure of one's OWN beliefs:
    - If my responses are correlated within domains (typical), transfer is stronger within domains
    - Gradients emerge as an artifact of self-response structure, not population statistics

    Uses the same threshold parameter as the Bayesian model for consistency.

    Parameters:
        obs_q: Index of observed question
        r_obs: Partner's observed response
        r_self: Self's responses (35,)
        base_rate: Base P(match)
        projection_weight: How much similarity boosts P(match)
        threshold: Scale for similarity decay (same as match_threshold in Bayesian model)

    Returns:
        (35,) array with question-specific predictions based on self-similarity
    """
    r_self_obs = r_self[obs_q]

    # GLOBAL: Perceived similarity from observed agreement
    # High when partner's response matches my response on observed question
    obs_diff = jnp.abs(r_obs - r_self_obs)
    perceived_similarity = jnp.exp(-obs_diff / threshold)

    # LOCAL: Self-response similarity for each question
    # High when I answered question q similarly to the observed question
    self_diff = jnp.abs(r_self - r_self_obs)
    self_similarity = jnp.exp(-self_diff / threshold)

    # Combined: projection modulated by self-similarity
    # "If you're like me (global), you'll answer like me on questions
    #  where my beliefs are consistent with the observed topic (local)"
    p_match = base_rate + perceived_similarity * self_similarity * projection_weight

    return jnp.clip(p_match, 0.01, 0.99)


@jit
def _predict_single(
    obs_q, r_obs, r_self, loadings, means,
    prior_cov, obs_variance, threshold,
    lambda_mix, base_rate, projection_weight
):
    """Combined prediction: (1-λ) × Bayesian + λ × SimilarityProjection."""
    p_bayes = _predict_bayesian(
        obs_q, r_obs, r_self, loadings, means,
        prior_cov, obs_variance, threshold
    )
    p_proj = _predict_similarity_projection(obs_q, r_obs, r_self, base_rate, projection_weight, threshold)

    return (1 - lambda_mix) * p_bayes + lambda_mix * p_proj


@jit
def _predict_batch(
    obs_qs, r_partners, r_selves, loadings, means,
    prior_cov, obs_variance, threshold,
    lambda_mix, base_rate, projection_weight
):
    """Batch prediction over participants using vmap."""
    return vmap(
        lambda oq, rp, rs: _predict_single(
            oq, rp, rs, loadings, means,
            prior_cov, obs_variance, threshold,
            lambda_mix, base_rate, projection_weight
        )
    )(obs_qs, r_partners, r_selves)


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

class CommonalityModel:
    """
    Commonality inference model combining Bayesian factor structure and
    similarity-modulated projection (Ames 2004; Tamir & Mitchell 2013).

    The mixture parameter λ controls the blend:
        P(match) = (1-λ) × Bayesian + λ × SimilarityProjection

    - λ=0: Pure Bayesian factor model (uses population structure)
    - λ=1: Pure similarity projection (uniform projection based on observed agreement)

    The Bayesian model predicts domain-specific gradients from factor structure.
    Similarity projection predicts UNIFORM shifts (no gradient) because
    perceived similarity is global and projection applies equally to all questions.

    Parameters
    ----------
    k : int
        Number of factors for Bayesian component (default 5)
    lambda_mix : float
        Mixture weight λ ∈ [0,1]. 0 = Bayesian, 1 = similarity projection.
    sigma_obs : float
        Observation noise standard deviation. Set to 0 for exact observations
        (no-chat condition) or >0 for soft observations (chat condition).
    sigma_prior : float
        Prior noise standard deviation (Bayesian model)
    match_threshold : float
        τ for defining a "match" (|r_partner - r_self| ≤ τ)
    epsilon : float
        Lapse rate (probability of random response)
    base_rate : float
        Base P(match) for similarity projection when perceived similarity = 0
    projection_weight : float
        How much perceived similarity boosts P(match) in similarity projection
    loadings : np.ndarray, optional
        Custom factor loadings (35 x k). If None, computed from data.
    question_means : np.ndarray, optional
        Custom question means (35,). If None, computed from data.
    """

    def __init__(
        self,
        k: int = 5,
        lambda_mix: float = 0.0,
        sigma_obs: float = 0.0,  # 0 = exact obs (no-chat), >0 = soft obs (chat)
        sigma_prior: float = 2.0,
        match_threshold: float = 1.5,
        epsilon: float = 0.2,
        base_rate: float = 0.3,
        projection_weight: float = 0.4,
        loadings: Optional[np.ndarray] = None,
        question_means: Optional[np.ndarray] = None,
    ):
        self.k = k
        self.lambda_mix = np.clip(lambda_mix, 0.0, 1.0)
        self.epsilon = np.clip(epsilon, 0.0, 1.0)
        self.base_rate = base_rate
        self.projection_weight = projection_weight

        # Load data
        means = question_means if question_means is not None else load_question_means()
        if k == 0:
            L = np.ones((N_QUESTIONS, 1))  # Flat: all questions identical
        elif loadings is not None:
            L = loadings[:, :k] if loadings.shape[1] > k else loadings
        else:
            L = load_factor_loadings(k)

        k_eff = L.shape[1]

        # Cache as JAX arrays
        self._loadings = jnp.array(L)
        self._means = jnp.array(means)
        self._prior_cov = jnp.array(sigma_prior**2 * np.eye(k_eff))
        self._obs_variance = jnp.array(sigma_obs**2)
        self._threshold = match_threshold
        self._lambda_mix = jnp.array(self.lambda_mix)
        self._base_rate = jnp.array(base_rate)
        self._projection_weight = jnp.array(projection_weight)

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
        preds = _predict_single(
            obs_q, r_partner, jnp.array(r_self),
            self._loadings, self._means,
            self._prior_cov, self._obs_variance, self._threshold,
            self._lambda_mix, self._base_rate, self._projection_weight
        )
        # Apply lapse rate
        preds = (1 - self.epsilon) * preds + self.epsilon * 0.5
        return np.asarray(preds)

    def predict_batch(self, obs_qs: np.ndarray, r_partners: np.ndarray,
                      r_selves: np.ndarray) -> np.ndarray:
        """
        Vectorized prediction for multiple participants.

        Args:
            obs_qs: (N,) array of observed question indices
            r_partners: (N,) array of partner responses
            r_selves: (N, 35) array of self responses

        Returns:
            (N, 35) array of match probabilities
        """
        preds = _predict_batch(
            jnp.array(obs_qs),
            jnp.array(r_partners),
            jnp.array(r_selves),
            self._loadings, self._means,
            self._prior_cov, self._obs_variance, self._threshold,
            self._lambda_mix, self._base_rate, self._projection_weight
        )
        # Apply lapse rate
        preds = (1 - self.epsilon) * preds + self.epsilon * 0.5
        return np.asarray(preds)

    def __repr__(self):
        if self.lambda_mix == 0:
            return f"CommonalityModel(k={self.k}, Bayesian)"
        elif self.lambda_mix == 1:
            return f"CommonalityModel(SimilarityProjection)"
        return f"CommonalityModel(k={self.k}, λ={self.lambda_mix:.2f})"


# =============================================================================
# FAST EVALUATION (for parameter fitting)
# =============================================================================

def prepare_evaluation_data(data: pd.DataFrame) -> dict:
    """
    Pre-compute arrays for fast batch evaluation.

    Returns dict with arrays ready for predict_batch.

    For the observed response (r_partner), we use:
    - Chat condition: postChatResponse (listener's perception of partner's response)
    - No-chat condition: partner_response (ground truth, directly observed)

    This is cognitively correct: the Bayesian update is based on what the
    listener believes they observed, not ground truth.
    """
    obs_qs, r_partners, r_selves = [], [], []
    participant_info = []  # (pid, question, domain, match_type, question_type, actual)

    for pid in data["pid"].unique():
        subj = data[data["pid"] == pid]
        matched = subj[subj["is_matched"] == True]
        if len(matched) == 0:
            continue

        obs_q = int(matched["matchedIdx"].iloc[0]) - 1

        # Use perceived response for chat, ground truth for no-chat
        experiment = subj["experiment"].iloc[0]
        if experiment == "chat":
            # Listener's perception of partner's response (from observed question row)
            obs_row = subj[subj["question_type"] == "observed"]
            if len(obs_row) == 0:
                continue
            r_partner = obs_row["postChatResponse"].iloc[0]
        else:
            # No-chat: direct observation, so use ground truth
            r_partner = matched["partner_response"].iloc[0]

        if pd.isna(r_partner):
            continue

        r_self = np.zeros(N_QUESTIONS)
        for _, row in subj.iterrows():
            r_self[int(row["question"]) - 1] = row["preChatResponse"]

        obs_qs.append(obs_q)
        r_partners.append(float(r_partner))
        r_selves.append(r_self)

        for _, row in subj.iterrows():
            participant_info.append({
                "pid": pid,  # Actual participant ID (for bootstrapping)
                "pid_idx": len(obs_qs) - 1,  # Index into batch arrays
                "question": int(row["question"]) - 1,
                "question_domain": row["preChatDomain"],
                "match_type": matched["match_type"].iloc[0],
                "question_type": row["question_type"],
                "actual": row["participant_binary_prediction"],
            })

    return {
        "obs_qs": np.array(obs_qs),
        "r_partners": np.array(r_partners),
        "r_selves": np.array(r_selves),
        "info": pd.DataFrame(participant_info),
    }


def compute_gradient_error(pred_df: pd.DataFrame, human_rates: dict) -> float:
    """
    Compute gradient error (deviation from human transfer effects).

    This is the metric minimized during parameter fitting.
    """
    model_rates = {}
    for qt in ['same_domain', 'different_domain']:
        for mt in ['high', 'low']:
            cell = pred_df[(pred_df["question_type"] == qt) & (pred_df["match_type"] == mt)]
            model_rates[(qt, mt)] = cell["pred_prob"].mean() if len(cell) else 0.5

    model_gradient = (model_rates[('same_domain', 'high')] - model_rates[('same_domain', 'low')]) - \
                     (model_rates[('different_domain', 'high')] - model_rates[('different_domain', 'low')])

    human_gradient = (human_rates[('same_domain', 'high')] - human_rates[('same_domain', 'low')]) - \
                     (human_rates[('different_domain', 'high')] - human_rates[('different_domain', 'low')])

    return abs(model_gradient - human_gradient)


def fit_parameters(
    k_values: list,
    eval_data: dict,
    human_rates: dict,
    lambda_mix: float = 0.0,
    verbose: bool = True,
) -> tuple[dict, dict]:
    """
    Fit model parameters by minimizing gradient error.

    Pass k_values=[5] for k=5-optimal params, or k_values=[1,2,3,...] for
    unified params that work across all k (fair for k-sweep comparison).

    Args:
        k_values: List of k values to optimize over (single k or multiple for unified)
        eval_data: Output of prepare_evaluation_data()
        human_rates: Dict of human rates {(question_type, match_type): rate}
        lambda_mix: Mixture weight λ ∈ [0,1]
        verbose: Print optimization progress

    Returns:
        (best_params, metrics) tuple

    Note: σ_obs is fixed at 0 for no-chat data since observations are exact
    (participants see explicit binary responses, not inferred from conversation).
    """
    from scipy.optimize import minimize

    # Only fit 3 params; σ_obs = 0 for exact observations (no-chat condition)
    bounds = {
        'sigma_prior': (0.1, 5.0),
        'match_threshold': (0.5, 4.0),
        'epsilon': (0.01, 0.5),
    }

    param_names = ['sigma_prior', 'match_threshold', 'epsilon']
    param_bounds = [bounds[p] for p in param_names]

    # Initialize with reasonable values
    x0 = np.array([1.5, 2.0, 0.1])

    n_evals = [0]
    def objective(x):
        n_evals[0] += 1
        sp, mt, eps = x
        total_error = 0.0
        for k in k_values:
            model = CommonalityModel(
                k=k, lambda_mix=lambda_mix,
                sigma_obs=0.0,  # Delta function for no-chat (exact observations)
                sigma_prior=sp, match_threshold=mt, epsilon=eps
            )
            pred_df = fast_evaluate(model, eval_data)
            total_error += compute_gradient_error(pred_df, human_rates)
        return total_error / len(k_values)  # Mean error across k

    if verbose:
        print(f"Optimizing params across k={k_values} (σ_obs=0 fixed)...")

    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        bounds=param_bounds,
        options={'ftol': 1e-8, 'gtol': 1e-6, 'maxiter': 200}
    )

    if verbose:
        print(f"  Converged: {result.success} after {n_evals[0]} evaluations")

    best_params = {'sigma_obs': 0.0}  # Delta function (exact observations in no-chat)
    best_params.update({p: float(result.x[i]) for i, p in enumerate(param_names)})

    # Compute per-k metrics
    metrics_by_k = {}
    for k in k_values:
        model = CommonalityModel(k=k, lambda_mix=lambda_mix, **best_params)
        pred_df = fast_evaluate(model, eval_data)
        probs = np.clip(pred_df["pred_prob"].values, 1e-10, 1-1e-10)
        actual = pred_df["actual"].values

        metrics_by_k[k] = {
            'gradient_error': compute_gradient_error(pred_df, human_rates),
            'log_likelihood': float(np.sum(actual * np.log(probs) + (1 - actual) * np.log(1 - probs))),
            'accuracy': float(np.mean((probs > 0.5) == actual)),
        }

    if verbose:
        print(f"  Final: σ_prior={best_params['sigma_prior']:.4f}, "
              f"τ={best_params['match_threshold']:.4f}, ε={best_params['epsilon']:.4f}")
        print(f"  Mean gradient error: {result.fun:.6f}")

    return best_params, {
        'mean_gradient_error': float(result.fun),
        'by_k': metrics_by_k,
        'optimizer_success': result.success,
    }


def fast_evaluate(model: CommonalityModel, eval_data: dict) -> pd.DataFrame:
    """
    Fast evaluation using batched predictions.

    Args:
        model: CommonalityModel instance
        eval_data: Output of prepare_evaluation_data()

    Returns:
        DataFrame with predictions
    """
    # Batch predict all participants at once
    all_preds = model.predict_batch(
        eval_data["obs_qs"],
        eval_data["r_partners"],
        eval_data["r_selves"]
    )

    # Map predictions back using numpy indexing (fast)
    info = eval_data["info"].copy()
    info["pred_prob"] = all_preds[
        info["pid_idx"].values,
        info["question"].values
    ]

    return info


