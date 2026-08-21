"""JAX/TFP translation of posterior_database/models/stan/dogs.stan:

    data { array[n_dogs, n_trials] int<lower=0, upper=1> y; }
    parameters { vector[3] beta; }
    transformed parameters {
      n_avoid[j,1] = 0; n_shock[j,1] = 0;
      n_avoid[j,t] = n_avoid[j,t-1] + 1 - y[j,t-1];
      n_shock[j,t] = n_shock[j,t-1] + y[j,t-1];
      p[j,t] = beta[1] + beta[2]*n_avoid[j,t] + beta[3]*n_shock[j,t];
    }
    model {
      beta ~ normal(0, 100);
      y[i,j] ~ bernoulli_logit(p[i,j]);
    }

`n_avoid`/`n_shock` are the same cumulative-count-before-t recursion as
`dogs_nonhierarchical`'s `_prev_shock_avoid` (reused here directly, under
this model's own variable names).
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


def _n_avoid_shock(y: jnp.ndarray):
    """y: (n_dogs, n_trials) binary. Returns n_avoid, n_shock: (n_dogs,
    n_trials), each column t holding the cumulative count of avoids/shocks
    strictly before trial t (column 0 is all zeros)."""
    j = y.shape[0]
    zeros_col = jnp.zeros((j, 1), dtype=y.dtype)
    n_shock = jnp.concatenate([zeros_col, jnp.cumsum(y[:, :-1], axis=1)], axis=1)
    n_avoid = jnp.concatenate([zeros_col, jnp.cumsum(1 - y[:, :-1], axis=1)], axis=1)
    return n_avoid, n_shock


class Dogs(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        return [ParameterSpec("beta", (3,), ConstraintType.UNCONSTRAINED)]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        beta = params["beta"]

        y = jnp.asarray(data["y"], dtype=jnp.float64)
        n_avoid, n_shock = _n_avoid_shock(y)
        p = beta[0] + beta[1] * n_avoid + beta[2] * n_shock

        lp = jnp.sum(tfd.Normal(loc=0.0, scale=100.0).log_prob(beta))
        lp = lp + jnp.sum(tfd.Bernoulli(logits=p).log_prob(y))
        return lp
