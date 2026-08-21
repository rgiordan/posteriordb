"""JAX/TFP translation of posterior_database/models/stan/radon_county.stan:

    parameters {
      vector[J] a; real mu_a;
      real<lower=0, upper=100> sigma_a;
      real<lower=0, upper=100> sigma_y;
    }
    model {
      y_hat[i] = a[county[i]];
      mu_a ~ normal(0, 1);
      a ~ normal(mu_a, sigma_a);
      y ~ normal(y_hat, sigma_y);
    }

`county` is Stan's 1-indexed group label; converted to a 0-indexed gather.
`sigma_a`/`sigma_y` have declared bounds but no explicit sampling statement,
so (per Stan's semantics) they contribute nothing beyond their bijector's own
Jacobian -- no extra term is added here.
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class RadonCounty(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        j = int(data["J"])
        return [
            ParameterSpec("a", (j,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("mu_a", (), ConstraintType.UNCONSTRAINED),
            ParameterSpec("sigma_a", (), ConstraintType.LOWER_UPPER, bounds=(0.0, 100.0)),
            ParameterSpec("sigma_y", (), ConstraintType.LOWER_UPPER, bounds=(0.0, 100.0)),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        a = params["a"]
        mu_a = params["mu_a"]
        sigma_a = params["sigma_a"]
        sigma_y = params["sigma_y"]

        county = jnp.asarray(data["county"]) - 1
        y_hat = a[county]

        lp = tfd.Normal(loc=0.0, scale=1.0).log_prob(mu_a)
        lp = lp + jnp.sum(tfd.Normal(loc=mu_a, scale=sigma_a).log_prob(a))
        lp = lp + jnp.sum(tfd.Normal(loc=y_hat, scale=sigma_y).log_prob(data["y"]))
        return lp
