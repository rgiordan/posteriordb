"""Shared code for the capture-recapture models (Kery & Schaub ch. 6): each
row of the augmented data set is either a real individual (present, with
probability omega) or a structural zero, so an individual never captured
contributes a two-component mixture."""

import jax
import jax.numpy as jnp


def eval_log_lik_augmented(*, omega, log_lik_captures, captured):
    """log p(captures) for one or more rows: `log_lik_captures` is the log
    probability of the capture history given the individual is present, and
    `captured` says whether it was ever caught."""
    present = jnp.log(omega) + log_lik_captures
    return jnp.where(captured, present, jnp.logaddexp(present, jnp.log1p(-omega)))


def eval_bernoulli_log_lik(*, y, p):
    """Summed Bernoulli log probabilities of a 0/1 capture history `y` (last axis)."""
    return jnp.sum(y * jnp.log(p) + (1 - y) * jnp.log1p(-p), axis=-1)


def eval_bernoulli_logit_log_lik(*, y, logit_p):
    return jnp.sum(y * logit_p - jax.nn.softplus(logit_p), axis=-1)


def sample_history(*, key, omega, p):
    """One augmented row: present with probability omega, then each occasion
    captured with probability p."""
    present_key, capture_key = jax.random.split(key)
    present = jax.random.bernoulli(present_key, p=omega)
    return (present & jax.random.bernoulli(capture_key, p=p)).astype(jnp.int32)
