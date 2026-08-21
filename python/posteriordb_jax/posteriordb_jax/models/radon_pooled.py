"""JAX/TFP translation of posterior_database/models/stan/radon_pooled.stan:

    parameters { real alpha; real beta; real<lower=0> sigma_y; }
    model {
      sigma_y ~ normal(0, 1);
      alpha ~ normal(0, 10);
      beta ~ normal(0, 10);
      mu = alpha + beta * floor_measure;
      log_radon ~ normal(mu, sigma_y);
    }
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class RadonPooled(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        return [
            ParameterSpec("alpha", (), ConstraintType.UNCONSTRAINED),
            ParameterSpec("beta", (), ConstraintType.UNCONSTRAINED),
            ParameterSpec("sigma_y", (), ConstraintType.LOWER, bounds=(0.0, None)),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        alpha = params["alpha"]
        beta = params["beta"]
        sigma_y = params["sigma_y"]

        mu = alpha + beta * data["floor_measure"]
        lp = tfd.Normal(loc=0.0, scale=1.0).log_prob(sigma_y)
        lp = lp + tfd.Normal(loc=0.0, scale=10.0).log_prob(alpha)
        lp = lp + tfd.Normal(loc=0.0, scale=10.0).log_prob(beta)
        lp = lp + jnp.sum(tfd.Normal(loc=mu, scale=sigma_y).log_prob(data["log_radon"]))
        return lp
