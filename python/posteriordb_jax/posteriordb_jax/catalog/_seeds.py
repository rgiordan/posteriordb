"""Shared likelihood of the three seeds models (germination counts on I
plates, each with its own random effect b)."""

import jax.numpy as jnp

from ._distributions import eval_binomial_logit_logpmf


def eval_log_lik(*, params, b, data):
    """n ~ binomial_logit(N, alpha0 + alpha1 * x1 + alpha2 * x2 + alpha12 * x1 .* x2 + b)."""
    x1, x2 = data["x1"], data["x2"]
    logit_p = (params["alpha0"] + params["alpha1"] * x1 + params["alpha2"] * x2
               + params["alpha12"] * x1 * x2 + b)
    return jnp.sum(eval_binomial_logit_logpmf(data["n"], n=data["N"], logit_p=logit_p))


def extract_data(raw):
    return {"n": jnp.asarray(raw["n"], dtype=jnp.int32), "N": jnp.asarray(raw["N"], dtype=jnp.int32),
            "x1": raw["x1"], "x2": raw["x2"]}
