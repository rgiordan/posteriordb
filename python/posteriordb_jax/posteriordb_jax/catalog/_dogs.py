"""Shared code for the dogs models (Solomon-Wynne avoidance learning): each
dog's 0/1 history y over trials, and its running counts of earlier shocks
(y = 1) and avoidances (y = 0)."""

import jax.numpy as jnp


def previous_counts(y):
    """(prev_shock, prev_avoid) before each trial, along the last axis of y."""
    shocks = jnp.cumsum(y, axis=-1) - y
    trials_before = jnp.arange(y.shape[-1])
    return shocks, trials_before - shocks


def bernoulli_log_lik(*, y, p):
    """Summed Bernoulli log probabilities. p is exactly 1 on each dog's first
    trial (no earlier shocks or avoidances), where y is 1, so log(1 - p) is
    only evaluated where y is 0 to keep values and gradients finite."""
    log_p = jnp.log(jnp.where(y == 1, p, 1.0))
    log_1mp = jnp.log1p(-jnp.where(y == 0, p, 0.0))
    return jnp.sum(jnp.where(y == 1, log_p, log_1mp), axis=-1)
