"""JAX/TFP translation of posterior_database/models/stan/seeds_model.stan:

    transformed data { vector[I] x1x2 = x1 .* x2; }
    parameters { real alpha0; real alpha1; real alpha12; real alpha2; real<lower=0> tau; vector[I] b; }
    transformed parameters { real<lower=0> sigma = 1.0 / sqrt(tau); }
    model {
      alpha0 ~ normal(0, 1e3); alpha1 ~ normal(0, 1e3);
      alpha2 ~ normal(0, 1e3); alpha12 ~ normal(0, 1e3);
      tau ~ gamma(1e-3, 1e-3);
      b ~ normal(0, sigma);
      n ~ binomial_logit(N, alpha0 + alpha1*x1 + alpha2*x2 + alpha12*x1x2 + b);
    }
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class SeedsModel(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        i = int(data["I"])
        return [
            ParameterSpec("alpha0", (), ConstraintType.UNCONSTRAINED),
            ParameterSpec("alpha1", (), ConstraintType.UNCONSTRAINED),
            ParameterSpec("alpha12", (), ConstraintType.UNCONSTRAINED),
            ParameterSpec("alpha2", (), ConstraintType.UNCONSTRAINED),
            ParameterSpec("tau", (), ConstraintType.LOWER, bounds=(0.0, None)),
            ParameterSpec("b", (i,), ConstraintType.UNCONSTRAINED),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        alpha0 = params["alpha0"]
        alpha1 = params["alpha1"]
        alpha12 = params["alpha12"]
        alpha2 = params["alpha2"]
        tau = params["tau"]
        b = params["b"]

        x1 = data["x1"]
        x2 = data["x2"]
        x1x2 = x1 * x2
        sigma = 1.0 / jnp.sqrt(tau)
        eta = alpha0 + alpha1 * x1 + alpha2 * x2 + alpha12 * x1x2 + b

        n = jnp.asarray(data["n"], dtype=jnp.float64)
        big_n = jnp.asarray(data["N"], dtype=jnp.float64)

        lp = tfd.Normal(loc=0.0, scale=1e3).log_prob(alpha0)
        lp = lp + tfd.Normal(loc=0.0, scale=1e3).log_prob(alpha1)
        lp = lp + tfd.Normal(loc=0.0, scale=1e3).log_prob(alpha2)
        lp = lp + tfd.Normal(loc=0.0, scale=1e3).log_prob(alpha12)
        lp = lp + tfd.Gamma(concentration=1e-3, rate=1e-3).log_prob(tau)
        lp = lp + jnp.sum(tfd.Normal(loc=0.0, scale=sigma).log_prob(b))
        lp = lp + jnp.sum(tfd.Binomial(total_count=big_n, logits=eta).log_prob(n))
        return lp
