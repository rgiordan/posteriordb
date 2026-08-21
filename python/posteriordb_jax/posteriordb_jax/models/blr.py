"""JAX/TFP translation of posterior_database/models/stan/blr.stan:

    data {
      int<lower=0> N;
      int<lower=0> D;
      matrix[N, D] X;
      vector[N] y;
    }
    parameters {
      vector[D] beta;
      real<lower=0> sigma;
    }
    model {
      target += normal_lpdf(beta | 0, 10);
      target += normal_lpdf(sigma | 0, 10);
      target += normal_lpdf(y | X * beta, sigma);
    }
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class Blr(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        d = int(data["D"])
        return [
            ParameterSpec("beta", (d,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("sigma", (), ConstraintType.LOWER, bounds=(0.0, None)),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        beta = params["beta"]
        sigma = params["sigma"]

        x = data["X"]
        y = data["y"]

        mu = x @ beta

        lp = jnp.sum(tfd.Normal(loc=0.0, scale=10.0).log_prob(beta))
        lp = lp + tfd.Normal(loc=0.0, scale=10.0).log_prob(sigma)
        lp = lp + jnp.sum(tfd.Normal(loc=mu, scale=sigma).log_prob(y))
        return lp
