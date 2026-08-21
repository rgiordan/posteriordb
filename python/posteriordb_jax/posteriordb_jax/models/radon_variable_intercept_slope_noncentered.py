"""JAX/TFP translation of posterior_database/models/stan/radon_variable_intercept_slope_noncentered.stan:

    parameters {
      real<lower=0> sigma_y; real<lower=0> sigma_alpha; real<lower=0> sigma_beta;
      vector[J] alpha_raw; vector[J] beta_raw; real mu_alpha; real mu_beta;
    }
    transformed parameters {
      vector[J] alpha = mu_alpha + sigma_alpha * alpha_raw;
      vector[J] beta = mu_beta + sigma_beta * beta_raw;
    }
    model {
      sigma_y ~ normal(0, 1); sigma_beta ~ normal(0, 1); sigma_alpha ~ normal(0, 1);
      mu_alpha ~ normal(0, 10); mu_beta ~ normal(0, 10);
      alpha_raw ~ normal(0, 1); beta_raw ~ normal(0, 1);
      mu[n] = alpha[county_idx[n]] + floor_measure[n] * beta[county_idx[n]];
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


class RadonVariableInterceptSlopeNoncentered(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        j = int(data["J"])
        return [
            ParameterSpec("sigma_y", (), ConstraintType.LOWER, bounds=(0.0, None)),
            ParameterSpec("sigma_alpha", (), ConstraintType.LOWER, bounds=(0.0, None)),
            ParameterSpec("sigma_beta", (), ConstraintType.LOWER, bounds=(0.0, None)),
            ParameterSpec("alpha_raw", (j,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("beta_raw", (j,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("mu_alpha", (), ConstraintType.UNCONSTRAINED),
            ParameterSpec("mu_beta", (), ConstraintType.UNCONSTRAINED),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        sigma_y = params["sigma_y"]
        sigma_alpha = params["sigma_alpha"]
        sigma_beta = params["sigma_beta"]
        alpha_raw = params["alpha_raw"]
        beta_raw = params["beta_raw"]
        mu_alpha = params["mu_alpha"]
        mu_beta = params["mu_beta"]

        alpha = mu_alpha + sigma_alpha * alpha_raw
        beta = mu_beta + sigma_beta * beta_raw
        county_idx = jnp.asarray(data["county_idx"]) - 1
        mu = alpha[county_idx] + data["floor_measure"] * beta[county_idx]

        lp = tfd.Normal(loc=0.0, scale=1.0).log_prob(sigma_y)
        lp = lp + tfd.Normal(loc=0.0, scale=1.0).log_prob(sigma_beta)
        lp = lp + tfd.Normal(loc=0.0, scale=1.0).log_prob(sigma_alpha)
        lp = lp + tfd.Normal(loc=0.0, scale=10.0).log_prob(mu_alpha)
        lp = lp + tfd.Normal(loc=0.0, scale=10.0).log_prob(mu_beta)
        lp = lp + jnp.sum(tfd.Normal(loc=0.0, scale=1.0).log_prob(alpha_raw))
        lp = lp + jnp.sum(tfd.Normal(loc=0.0, scale=1.0).log_prob(beta_raw))
        lp = lp + jnp.sum(tfd.Normal(loc=mu, scale=sigma_y).log_prob(data["log_radon"]))
        return lp
