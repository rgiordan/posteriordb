"""JAX/TFP translation of posterior_database/models/stan/pilots.stan:

    parameters {
      vector[n_groups] a; vector[n_scenarios] b; real mu_a; real mu_b;
      real<lower=0, upper=100> sigma_a; real<lower=0, upper=100> sigma_b; real<lower=0, upper=100> sigma_y;
    }
    transformed parameters { y_hat[i] = a[group_id[i]] + b[scenario_id[i]]; }
    model {
      mu_a ~ normal(0, 1); a ~ normal(10 * mu_a, sigma_a);
      mu_b ~ normal(0, 1); b ~ normal(10 * mu_b, sigma_b);
      y ~ normal(y_hat, sigma_y);
    }
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class Pilots(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        n_groups = int(data["n_groups"])
        n_scenarios = int(data["n_scenarios"])
        return [
            ParameterSpec("a", (n_groups,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("b", (n_scenarios,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("mu_a", (), ConstraintType.UNCONSTRAINED),
            ParameterSpec("mu_b", (), ConstraintType.UNCONSTRAINED),
            ParameterSpec("sigma_a", (), ConstraintType.LOWER_UPPER, bounds=(0.0, 100.0)),
            ParameterSpec("sigma_b", (), ConstraintType.LOWER_UPPER, bounds=(0.0, 100.0)),
            ParameterSpec("sigma_y", (), ConstraintType.LOWER_UPPER, bounds=(0.0, 100.0)),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        a = params["a"]
        b = params["b"]
        mu_a = params["mu_a"]
        mu_b = params["mu_b"]
        sigma_a = params["sigma_a"]
        sigma_b = params["sigma_b"]
        sigma_y = params["sigma_y"]

        group_id = jnp.asarray(data["group_id"]) - 1
        scenario_id = jnp.asarray(data["scenario_id"]) - 1
        y_hat = a[group_id] + b[scenario_id]

        lp = tfd.Normal(loc=0.0, scale=1.0).log_prob(mu_a)
        lp = lp + jnp.sum(tfd.Normal(loc=10.0 * mu_a, scale=sigma_a).log_prob(a))
        lp = lp + tfd.Normal(loc=0.0, scale=1.0).log_prob(mu_b)
        lp = lp + jnp.sum(tfd.Normal(loc=10.0 * mu_b, scale=sigma_b).log_prob(b))
        lp = lp + jnp.sum(tfd.Normal(loc=y_hat, scale=sigma_y).log_prob(data["y"]))
        return lp
