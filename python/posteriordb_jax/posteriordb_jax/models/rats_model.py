"""JAX/TFP translation of posterior_database/models/stan/rats_model.stan:

    parameters {
      array[N] real alpha; array[N] real beta;
      real mu_alpha; real mu_beta;
      real<lower=0> sigma_y; real<lower=0> sigma_alpha; real<lower=0> sigma_beta;
    }
    model {
      mu_alpha ~ normal(0, 100); mu_beta ~ normal(0, 100);
      alpha ~ normal(mu_alpha, sigma_alpha);
      beta ~ normal(mu_beta, sigma_beta);
      y[n] ~ normal(alpha[rat[n]] + beta[rat[n]] * (x[n] - xbar), sigma_y);
    }

(`generated quantities`'s `alpha0` is a derived, RNG-free quantity that
doesn't affect the log density and is omitted.)
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class RatsModel(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        n = int(data["N"])
        return [
            ParameterSpec("alpha", (n,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("beta", (n,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("mu_alpha", (), ConstraintType.UNCONSTRAINED),
            ParameterSpec("mu_beta", (), ConstraintType.UNCONSTRAINED),
            ParameterSpec("sigma_y", (), ConstraintType.LOWER, bounds=(0.0, None)),
            ParameterSpec("sigma_alpha", (), ConstraintType.LOWER, bounds=(0.0, None)),
            ParameterSpec("sigma_beta", (), ConstraintType.LOWER, bounds=(0.0, None)),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        alpha = params["alpha"]
        beta = params["beta"]
        mu_alpha = params["mu_alpha"]
        mu_beta = params["mu_beta"]
        sigma_y = params["sigma_y"]
        sigma_alpha = params["sigma_alpha"]
        sigma_beta = params["sigma_beta"]

        rat = jnp.asarray(data["rat"]) - 1
        x = data["x"]
        xbar = data["xbar"]
        mu = alpha[rat] + beta[rat] * (x - xbar)

        lp = tfd.Normal(loc=0.0, scale=100.0).log_prob(mu_alpha)
        lp = lp + tfd.Normal(loc=0.0, scale=100.0).log_prob(mu_beta)
        lp = lp + jnp.sum(tfd.Normal(loc=mu_alpha, scale=sigma_alpha).log_prob(alpha))
        lp = lp + jnp.sum(tfd.Normal(loc=mu_beta, scale=sigma_beta).log_prob(beta))
        lp = lp + jnp.sum(tfd.Normal(loc=mu, scale=sigma_y).log_prob(data["y"]))
        return lp
