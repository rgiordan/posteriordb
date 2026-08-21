"""JAX/TFP translation of posterior_database/models/stan/irt_2pl.stan:

    parameters {
      real<lower=0> sigma_theta; vector[J] theta;
      real<lower=0> sigma_a; vector<lower=0>[I] a;
      real mu_b; real<lower=0> sigma_b; vector[I] b;
    }
    model {
      sigma_theta ~ cauchy(0, 2); theta ~ normal(0, sigma_theta);
      sigma_a ~ cauchy(0, 2); a ~ lognormal(0, sigma_a);
      mu_b ~ normal(0, 5); sigma_b ~ cauchy(0, 2); b ~ normal(mu_b, sigma_b);
      for (i in 1:I) { y[i] ~ bernoulli_logit(a[i] * (theta - b[i])); }
    }
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class Irt2pl(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        i = int(data["I"])
        j = int(data["J"])
        return [
            ParameterSpec("sigma_theta", (), ConstraintType.LOWER, bounds=(0.0, None)),
            ParameterSpec("theta", (j,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("sigma_a", (), ConstraintType.LOWER, bounds=(0.0, None)),
            ParameterSpec("a", (i,), ConstraintType.LOWER, bounds=(0.0, None)),
            ParameterSpec("mu_b", (), ConstraintType.UNCONSTRAINED),
            ParameterSpec("sigma_b", (), ConstraintType.LOWER, bounds=(0.0, None)),
            ParameterSpec("b", (i,), ConstraintType.UNCONSTRAINED),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        sigma_theta = params["sigma_theta"]
        theta = params["theta"]
        sigma_a = params["sigma_a"]
        a = params["a"]
        mu_b = params["mu_b"]
        sigma_b = params["sigma_b"]
        b = params["b"]

        y = jnp.asarray(data["y"], dtype=jnp.float64)  # (I, J)
        logits = a[:, None] * (theta[None, :] - b[:, None])  # (I, J)

        lp = tfd.Cauchy(loc=0.0, scale=2.0).log_prob(sigma_theta)
        lp = lp + jnp.sum(tfd.Normal(loc=0.0, scale=sigma_theta).log_prob(theta))
        lp = lp + tfd.Cauchy(loc=0.0, scale=2.0).log_prob(sigma_a)
        lp = lp + jnp.sum(tfd.LogNormal(loc=0.0, scale=sigma_a).log_prob(a))
        lp = lp + tfd.Normal(loc=0.0, scale=5.0).log_prob(mu_b)
        lp = lp + tfd.Cauchy(loc=0.0, scale=2.0).log_prob(sigma_b)
        lp = lp + jnp.sum(tfd.Normal(loc=mu_b, scale=sigma_b).log_prob(b))
        lp = lp + jnp.sum(tfd.Bernoulli(logits=logits).log_prob(y))
        return lp
