"""JAX/TFP translation of posterior_database/models/stan/mesquite.stan:

    data {
      int<lower=0> N;
      vector[N] weight;
      vector[N] diam1;
      vector[N] diam2;
      vector[N] canopy_height;
      vector[N] total_height;
      vector[N] density;
      vector[N] group;
    }
    parameters {
      vector[7] beta;
      real<lower=0> sigma;
    }
    model {
      weight ~ normal(beta[1] + beta[2] * diam1 + beta[3] * diam2
                      + beta[4] * canopy_height + beta[5] * total_height
                      + beta[6] * density + beta[7] * group, sigma);
    }
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class Mesquite(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        return [
            ParameterSpec("beta", (7,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("sigma", (), ConstraintType.LOWER, bounds=(0.0, None)),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        beta = params["beta"]
        sigma = params["sigma"]

        mu = (
            beta[0]
            + beta[1] * data["diam1"]
            + beta[2] * data["diam2"]
            + beta[3] * data["canopy_height"]
            + beta[4] * data["total_height"]
            + beta[5] * data["density"]
            + beta[6] * data["group"]
        )
        return jnp.sum(tfd.Normal(loc=mu, scale=sigma).log_prob(data["weight"]))
