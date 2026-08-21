"""JAX/TFP translation of posterior_database/models/stan/logearn_interaction_z.stan:

    transformed data {
      vector[N] log_earn = log(earn);
      vector[N] z_height = (height - mean(height)) / sd(height);
      vector[N] inter = z_height .* male;
    }
    parameters { vector[4] beta; real<lower=0> sigma; }
    model {
      log_earn ~ normal(beta[1] + beta[2] * z_height + beta[3] * male
                        + beta[4] * inter, sigma);
    }

Stan's `sd()` is the sample standard deviation (`ddof=1`).
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class LogearnInteractionZ(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        return [
            ParameterSpec("beta", (4,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("sigma", (), ConstraintType.LOWER, bounds=(0.0, None)),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        beta = params["beta"]
        sigma = params["sigma"]
        height = data["height"]
        male = data["male"]
        log_earn = jnp.log(data["earn"])
        z_height = (height - jnp.mean(height)) / jnp.std(height, ddof=1)
        inter = z_height * male
        mu = beta[0] + beta[1] * z_height + beta[2] * male + beta[3] * inter
        return jnp.sum(tfd.Normal(loc=mu, scale=sigma).log_prob(log_earn))
