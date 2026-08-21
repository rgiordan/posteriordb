"""JAX/TFP translation of posterior_database/models/stan/logearn_height_male.stan:

    transformed data { vector[N] log_earn = log(earn); }
    parameters { vector[3] beta; real<lower=0> sigma; }
    model { log_earn ~ normal(beta[1] + beta[2] * height + beta[3] * male, sigma); }
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class LogearnHeightMale(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        return [
            ParameterSpec("beta", (3,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("sigma", (), ConstraintType.LOWER, bounds=(0.0, None)),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        beta = params["beta"]
        sigma = params["sigma"]
        log_earn = jnp.log(data["earn"])
        mu = beta[0] + beta[1] * data["height"] + beta[2] * data["male"]
        return jnp.sum(tfd.Normal(loc=mu, scale=sigma).log_prob(log_earn))
