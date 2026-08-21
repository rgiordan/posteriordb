"""JAX/TFP translation of posterior_database/models/stan/radon_county_intercept.stan:

    parameters { vector[J] alpha; real beta; real<lower=0> sigma_y; }
    model {
      sigma_y ~ normal(0, 1);
      alpha ~ normal(0, 10);
      beta ~ normal(0, 10);
      mu[n] = alpha[county_idx[n]] + beta * floor_measure[n];
      log_radon ~ normal(mu, sigma_y);
    }

`county_idx` is Stan's 1-indexed group label; converted to a 0-indexed gather.
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class RadonCountyIntercept(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        j = int(data["J"])
        return [
            ParameterSpec("alpha", (j,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("beta", (), ConstraintType.UNCONSTRAINED),
            ParameterSpec("sigma_y", (), ConstraintType.LOWER, bounds=(0.0, None)),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        alpha = params["alpha"]
        beta = params["beta"]
        sigma_y = params["sigma_y"]

        county_idx = jnp.asarray(data["county_idx"]) - 1
        mu = alpha[county_idx] + beta * data["floor_measure"]

        lp = tfd.Normal(loc=0.0, scale=1.0).log_prob(sigma_y)
        lp = lp + jnp.sum(tfd.Normal(loc=0.0, scale=10.0).log_prob(alpha))
        lp = lp + tfd.Normal(loc=0.0, scale=10.0).log_prob(beta)
        lp = lp + jnp.sum(tfd.Normal(loc=mu, scale=sigma_y).log_prob(data["log_radon"]))
        return lp
