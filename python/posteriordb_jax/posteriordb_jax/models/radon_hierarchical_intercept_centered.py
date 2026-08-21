"""JAX/TFP translation of posterior_database/models/stan/radon_hierarchical_intercept_centered.stan:

    parameters { vector[J] alpha; vector[2] beta; real mu_alpha; real<lower=0> sigma_alpha; real<lower=0> sigma_y; }
    model {
      sigma_alpha ~ normal(0, 1); sigma_y ~ normal(0, 1);
      mu_alpha ~ normal(0, 10); beta ~ normal(0, 10);
      alpha ~ normal(mu_alpha, sigma_alpha);
      muj[n] = alpha[county_idx[n]] + log_uppm[n] * beta[1];
      mu[n] = muj[n] + floor_measure[n] * beta[2];
      log_radon[n] ~ normal(mu[n], sigma_y);
    }
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class RadonHierarchicalInterceptCentered(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        j = int(data["J"])
        return [
            ParameterSpec("alpha", (j,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("beta", (2,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("mu_alpha", (), ConstraintType.UNCONSTRAINED),
            ParameterSpec("sigma_alpha", (), ConstraintType.LOWER, bounds=(0.0, None)),
            ParameterSpec("sigma_y", (), ConstraintType.LOWER, bounds=(0.0, None)),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        alpha = params["alpha"]
        beta = params["beta"]
        mu_alpha = params["mu_alpha"]
        sigma_alpha = params["sigma_alpha"]
        sigma_y = params["sigma_y"]

        county_idx = jnp.asarray(data["county_idx"]) - 1
        muj = alpha[county_idx] + data["log_uppm"] * beta[0]
        mu = muj + data["floor_measure"] * beta[1]

        lp = tfd.Normal(loc=0.0, scale=1.0).log_prob(sigma_alpha)
        lp = lp + tfd.Normal(loc=0.0, scale=1.0).log_prob(sigma_y)
        lp = lp + tfd.Normal(loc=0.0, scale=10.0).log_prob(mu_alpha)
        lp = lp + jnp.sum(tfd.Normal(loc=0.0, scale=10.0).log_prob(beta))
        lp = lp + jnp.sum(tfd.Normal(loc=mu_alpha, scale=sigma_alpha).log_prob(alpha))
        lp = lp + jnp.sum(tfd.Normal(loc=mu, scale=sigma_y).log_prob(data["log_radon"]))
        return lp
