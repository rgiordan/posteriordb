"""JAX/TFP translation of posterior_database/models/stan/logmesquite_logvash.stan:

    transformed data {
      log_weight = log(weight);
      log_canopy_volume = log(diam1 .* diam2 .* canopy_height);
      log_canopy_area = log(diam1 .* diam2);
      log_canopy_shape = log(diam1 ./ diam2);
      log_total_height = log(total_height);
    }
    parameters { vector[6] beta; real<lower=0> sigma; }
    model {
      log_weight ~ normal(beta[1] + beta[2] * log_canopy_volume
                          + beta[3] * log_canopy_area
                          + beta[4] * log_canopy_shape
                          + beta[5] * log_total_height + beta[6] * group, sigma);
    }
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class LogmesquiteLogvash(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        return [
            ParameterSpec("beta", (6,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("sigma", (), ConstraintType.LOWER, bounds=(0.0, None)),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        beta = params["beta"]
        sigma = params["sigma"]
        diam1 = data["diam1"]
        diam2 = data["diam2"]
        log_weight = jnp.log(data["weight"])
        log_canopy_volume = jnp.log(diam1 * diam2 * data["canopy_height"])
        log_canopy_area = jnp.log(diam1 * diam2)
        log_canopy_shape = jnp.log(diam1 / diam2)
        log_total_height = jnp.log(data["total_height"])
        mu = (
            beta[0]
            + beta[1] * log_canopy_volume
            + beta[2] * log_canopy_area
            + beta[3] * log_canopy_shape
            + beta[4] * log_total_height
            + beta[5] * data["group"]
        )
        return jnp.sum(tfd.Normal(loc=mu, scale=sigma).log_prob(log_weight))
