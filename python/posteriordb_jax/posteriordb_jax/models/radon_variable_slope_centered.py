"""JAX/TFP translation of posterior_database/models/stan/radon_variable_slope_centered.stan:

    parameters { real alpha; vector[J] beta; real mu_beta; real<lower=0> sigma_beta; real<lower=0> sigma_y; }
    model {
      alpha ~ normal(0, 10); sigma_y ~ normal(0, 1);
      sigma_beta ~ normal(0, 1); mu_beta ~ normal(0, 10);
      beta ~ normal(mu_beta, sigma_beta);
      mu[n] = alpha + floor_measure[n] * beta[county_idx[n]];
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


class RadonVariableSlopeCentered(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        j = int(data["J"])
        return [
            ParameterSpec("alpha", (), ConstraintType.UNCONSTRAINED),
            ParameterSpec("beta", (j,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("mu_beta", (), ConstraintType.UNCONSTRAINED),
            ParameterSpec("sigma_beta", (), ConstraintType.LOWER, bounds=(0.0, None)),
            ParameterSpec("sigma_y", (), ConstraintType.LOWER, bounds=(0.0, None)),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        alpha = params["alpha"]
        beta = params["beta"]
        mu_beta = params["mu_beta"]
        sigma_beta = params["sigma_beta"]
        sigma_y = params["sigma_y"]

        county_idx = jnp.asarray(data["county_idx"]) - 1
        mu = alpha + data["floor_measure"] * beta[county_idx]

        lp = tfd.Normal(loc=0.0, scale=10.0).log_prob(alpha)
        lp = lp + tfd.Normal(loc=0.0, scale=1.0).log_prob(sigma_y)
        lp = lp + tfd.Normal(loc=0.0, scale=1.0).log_prob(sigma_beta)
        lp = lp + tfd.Normal(loc=0.0, scale=10.0).log_prob(mu_beta)
        lp = lp + jnp.sum(tfd.Normal(loc=mu_beta, scale=sigma_beta).log_prob(beta))
        lp = lp + jnp.sum(tfd.Normal(loc=mu, scale=sigma_y).log_prob(data["log_radon"]))
        return lp
