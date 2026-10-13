"""The forward algorithm shared by the hidden Markov models."""

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


def eval_forward_log_marginal(*, log_transition, log_emission, log_initial=None):
    """log p(y_1:T), summing over the hidden states with the forward
    recursion gamma[t, k] = log_sum_exp_j(gamma[t - 1, j] + log_transition[j, k])
    + log_emission[t, k]. `log_emission` is (T, K); with no `log_initial`,
    gamma[1] = log_emission[1], as in Stan models that omit an initial
    distribution."""
    gamma_first = log_emission[0] if log_initial is None else log_initial + log_emission[0]

    def step(gamma_prev, log_emission_t):
        return logsumexp(gamma_prev[:, None] + log_transition, axis=0) + log_emission_t, None

    gamma_last, _ = jax.lax.scan(step, gamma_first, log_emission[1:])
    return logsumexp(gamma_last)
