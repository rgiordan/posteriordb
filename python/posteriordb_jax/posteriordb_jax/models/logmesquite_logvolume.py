"""JAX/TFP translation of posterior_database/models/stan/logmesquite_logvolume.stan:

    transformed data {
      log_weight = log(weight);
      log_canopy_volume = log(diam1 .* diam2 .* canopy_height);
    }
    parameters { vector[2] beta; real<lower=0> sigma; }
    model { log_weight ~ normal(beta[1] + beta[2] * log_canopy_volume, sigma); }
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class LogmesquiteLogvolume(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        return [
            ParameterSpec("beta", (2,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("sigma", (), ConstraintType.LOWER, bounds=(0.0, None)),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        beta = params["beta"]
        sigma = params["sigma"]
        log_weight = jnp.log(data["weight"])
        log_canopy_volume = jnp.log(data["diam1"] * data["diam2"] * data["canopy_height"])
        mu = beta[0] + beta[1] * log_canopy_volume
        return jnp.sum(tfd.Normal(loc=mu, scale=sigma).log_prob(log_weight))
