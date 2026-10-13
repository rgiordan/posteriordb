"""Pieces shared by the latent-regression item response models
(2pl_latent_reg_irt, gpcm_latent_reg_irt, grsm_latent_reg_irt): the
covariate adjustments, the sum-to-zero difficulties, the common priors, and
the regression of ability theta on the adjusted covariates."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy import stats

from ._distributions import eval_lognormal_logpdf


def obtain_adjustments(w):
    """Stan's obtain_adjustments, including its operator precedence:
    `minmax_count + W[j, k] == min_w || W[j, k] == max_w` parses as
    `((minmax_count + W[j, k]) == min_w) || (W[j, k] == max_w)`, so in
    practice the two-valued branch is taken only when rows(W) == 1."""
    n_rows, n_cols = w.shape
    adj = np.zeros((2, n_cols))
    adj[1, 0] = 1.0
    for k in range(1, n_cols):
        column = w[:, k]
        min_w, max_w = column.min(), column.max()
        minmax_count = 0
        for value in column:
            minmax_count = int((minmax_count + value) == min_w or value == max_w)
        adj[0, k] = column.mean()
        adj[1, k] = max_w - min_w if minmax_count == n_rows else 2.0 * column.std(ddof=1)
    return adj


def adjust_covariates(w):
    """W_adj = (W - adj[1]) ./ adj[2], as a jnp array."""
    w = np.asarray(w, dtype=float)
    adj = obtain_adjustments(w)
    return jnp.asarray((w - adj[0]) / adj[1])


def sum_to_zero(free):
    """append_row(free, -sum(free))."""
    return jnp.append(free, -jnp.sum(free))


def eval_log_prior_alpha_lambda(params):
    """alpha ~ lognormal(1, 1); lambda_adj ~ student_t(3, 0, 1)."""
    return (jnp.sum(eval_lognormal_logpdf(params["alpha"], mu=1.0, sigma=1.0))
            + jnp.sum(stats.t.logpdf(params["lambda_adj"], df=3.0, loc=0.0, scale=1.0)))


def eval_theta_log_density(*, params, data):
    """theta ~ normal(W_adj * lambda_adj, 1)."""
    return jnp.sum(stats.norm.logpdf(params["theta"], loc=data["w_adj"] @ params["lambda_adj"], scale=1.0))


def eval_categorical_steps_logpmf(*, y, steps, mask=None):
    """categorical_lpmf(y + 1 | softmax(cumulative_sum(append_row(0, steps))))
    for each row of `steps` (N, M). Entries where `mask` is False are padding
    (they must come last in each row) and are left out."""
    if mask is not None:
        steps = jnp.where(mask, steps, 0.0)
    cumulative = jnp.concatenate([jnp.zeros((steps.shape[0], 1)), jnp.cumsum(steps, axis=1)], axis=1)
    if mask is not None:
        valid = jnp.concatenate([jnp.ones((steps.shape[0], 1), dtype=bool), mask], axis=1)
        cumulative_for_norm = jnp.where(valid, cumulative, -jnp.inf)
    else:
        cumulative_for_norm = cumulative
    log_norm = jax.scipy.special.logsumexp(cumulative_for_norm, axis=1)
    return jnp.take_along_axis(cumulative, y[:, None], axis=1)[:, 0] - log_norm
