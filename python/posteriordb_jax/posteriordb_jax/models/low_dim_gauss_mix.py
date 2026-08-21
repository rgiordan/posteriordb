"""JAX/TFP translation of posterior_database/models/stan/low_dim_gauss_mix.stan:

    data {
      int<lower=0> N;
      vector[N] y;
    }
    parameters {
      ordered[2] mu;
      array[2] real<lower=0> sigma;
      real<lower=0, upper=1> theta;
    }
    model {
      sigma ~ normal(0, 2);
      mu ~ normal(0, 2);
      theta ~ beta(5, 5);
      for (n in 1 : N) {
        target += log_mix(theta, normal_lpdf(y[n] | mu[1], sigma[1]),
                          normal_lpdf(y[n] | mu[2], sigma[2]));
      }
    }
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class LowDimGaussMix(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        return [
            ParameterSpec("mu", (2,), ConstraintType.ORDERED),
            ParameterSpec("sigma", (2,), ConstraintType.LOWER, bounds=(0.0, None)),
            ParameterSpec("theta", (), ConstraintType.LOWER_UPPER, bounds=(0.0, 1.0)),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        mu = params["mu"]
        sigma = params["sigma"]
        theta = params["theta"]

        y = data["y"]

        lp = jnp.sum(tfd.Normal(loc=0.0, scale=2.0).log_prob(sigma))
        lp = lp + jnp.sum(tfd.Normal(loc=0.0, scale=2.0).log_prob(mu))
        lp = lp + tfd.Beta(concentration1=5.0, concentration0=5.0).log_prob(theta)

        lp1 = tfd.Normal(loc=mu[0], scale=sigma[0]).log_prob(y)
        lp2 = tfd.Normal(loc=mu[1], scale=sigma[1]).log_prob(y)
        # log_mix(theta, lp1, lp2) = log(theta*exp(lp1) + (1-theta)*exp(lp2))
        mix = jnp.logaddexp(jnp.log(theta) + lp1, jnp.log1p(-theta) + lp2)
        lp = lp + jnp.sum(mix)
        return lp
