"""JAX/TFP translation of posterior_database/models/stan/lsat_model.stan:

    data { array[R] int<lower=0> culm; array[R, T] int<lower=0> response; }
    transformed data {
      // r[T, N]: expands each of the R response *patterns* into
      // (culm[i]-culm[i-1]) individual students' per-item responses.
    }
    parameters { array[T] real alpha; vector[N] theta; real<lower=0> beta; }
    model {
      alpha ~ normal(0, 100); theta ~ normal(0, 1); beta ~ normal(0, 100);
      r[k] ~ bernoulli_logit(beta * theta - alpha[k] * ones);
    }

`response`/`culm` is a run-length encoding: `culm[i]` is the cumulative
student count with response pattern `<= i`, so `response[i, :]` is shared
by students `culm[i-1]+1 .. culm[i]`. Expanded here with `jnp.repeat` into
a per-student `(N, T)` array (data-only, no parameters involved, so the
repeat counts are always static w.r.t. anything being differentiated).
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


def _expand_responses(culm, response, n: int):
    culm = jnp.asarray(culm)
    counts = culm - jnp.concatenate([jnp.zeros((1,), dtype=culm.dtype), culm[:-1]])
    response = jnp.asarray(response, dtype=jnp.float64)  # (R, T)
    return jnp.repeat(response, counts, axis=0, total_repeat_length=n)  # (N, T)


class LsatModel(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        t = int(data["T"])
        n = int(data["N"])
        return [
            ParameterSpec("alpha", (t,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("theta", (n,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("beta", (), ConstraintType.LOWER, bounds=(0.0, None)),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        alpha = params["alpha"]
        theta = params["theta"]
        beta = params["beta"]

        n = int(data["N"])
        r = _expand_responses(data["culm"], data["response"], n)  # (N, T)
        logits = beta * theta[:, None] - alpha[None, :]  # (N, T)

        lp = jnp.sum(tfd.Normal(loc=0.0, scale=100.0).log_prob(alpha))
        lp = lp + jnp.sum(tfd.Normal(loc=0.0, scale=1.0).log_prob(theta))
        lp = lp + tfd.Normal(loc=0.0, scale=100.0).log_prob(beta)
        lp = lp + jnp.sum(tfd.Bernoulli(logits=logits).log_prob(r))
        return lp
