"""JAX/TFP translation of posterior_database/models/stan/radon_variable_intercept_noncentered.stan:

    parameters { vector[J] alpha_raw; real beta; real mu_alpha; real<lower=0> sigma_alpha; real<lower=0> sigma_y; }
    transformed parameters { vector[J] alpha = mu_alpha + sigma_alpha * alpha_raw; }
    model {
      sigma_y ~ normal(0, 1); sigma_alpha ~ normal(0, 1);
      mu_alpha ~ normal(0, 10); beta ~ normal(0, 10); alpha_raw ~ normal(0, 1);
      mu[n] = alpha[county_idx[n]] + floor_measure[n] * beta;
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


class RadonVariableInterceptNoncentered(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        j = int(data["J"])
        return [
            ParameterSpec("alpha_raw", (j,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("beta", (), ConstraintType.UNCONSTRAINED),
            ParameterSpec("mu_alpha", (), ConstraintType.UNCONSTRAINED),
            ParameterSpec("sigma_alpha", (), ConstraintType.LOWER, bounds=(0.0, None)),
            ParameterSpec("sigma_y", (), ConstraintType.LOWER, bounds=(0.0, None)),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        alpha_raw = params["alpha_raw"]
        beta = params["beta"]
        mu_alpha = params["mu_alpha"]
        sigma_alpha = params["sigma_alpha"]
        sigma_y = params["sigma_y"]

        alpha = mu_alpha + sigma_alpha * alpha_raw
        county_idx = jnp.asarray(data["county_idx"]) - 1
        mu = alpha[county_idx] + data["floor_measure"] * beta

        lp = tfd.Normal(loc=0.0, scale=1.0).log_prob(sigma_y)
        lp = lp + tfd.Normal(loc=0.0, scale=1.0).log_prob(sigma_alpha)
        lp = lp + tfd.Normal(loc=0.0, scale=10.0).log_prob(mu_alpha)
        lp = lp + tfd.Normal(loc=0.0, scale=10.0).log_prob(beta)
        lp = lp + jnp.sum(tfd.Normal(loc=0.0, scale=1.0).log_prob(alpha_raw))
        lp = lp + jnp.sum(tfd.Normal(loc=mu, scale=sigma_y).log_prob(data["log_radon"]))
        return lp
