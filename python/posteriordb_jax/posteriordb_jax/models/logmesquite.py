"""JAX/TFP translation of posterior_database/models/stan/logmesquite.stan:

    transformed data { log-transform weight, diam1, diam2, canopy_height, total_height, density; }
    parameters { vector[7] beta; real<lower=0> sigma; }
    model {
      log_weight ~ normal(beta[1] + beta[2] * log_diam1 + beta[3] * log_diam2
                          + beta[4] * log_canopy_height
                          + beta[5] * log_total_height + beta[6] * log_density
                          + beta[7] * group, sigma);
    }
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class Logmesquite(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        return [
            ParameterSpec("beta", (7,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("sigma", (), ConstraintType.LOWER, bounds=(0.0, None)),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        beta = params["beta"]
        sigma = params["sigma"]
        log_weight = jnp.log(data["weight"])
        log_diam1 = jnp.log(data["diam1"])
        log_diam2 = jnp.log(data["diam2"])
        log_canopy_height = jnp.log(data["canopy_height"])
        log_total_height = jnp.log(data["total_height"])
        log_density_ = jnp.log(data["density"])
        mu = (
            beta[0]
            + beta[1] * log_diam1
            + beta[2] * log_diam2
            + beta[3] * log_canopy_height
            + beta[4] * log_total_height
            + beta[5] * log_density_
            + beta[6] * data["group"]
        )
        return jnp.sum(tfd.Normal(loc=mu, scale=sigma).log_prob(log_weight))
