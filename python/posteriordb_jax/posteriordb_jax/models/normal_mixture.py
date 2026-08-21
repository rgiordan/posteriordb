"""JAX/TFP translation of posterior_database/models/stan/normal_mixture.stan:

    parameters { real<lower=0, upper=1> theta; array[2] real mu; }
    model {
      theta ~ uniform(0, 1);
      for (k in 1 : 2) { mu[k] ~ normal(0, 10); }
      for (n in 1 : N) {
        target += log_mix(theta, normal_lpdf(y[n] | mu[1], 1.0),
                          normal_lpdf(y[n] | mu[2], 1.0));
      }
    }

`theta ~ uniform(0, 1)` is an explicit sampling statement even though it's a
no-op *density-value* contribution only because the bounds happen to span a
unit interval (`uniform_lpdf` on `[0,1]` is exactly 0); included via
`tfd.Uniform` for a literal translation.
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class NormalMixture(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        return [
            ParameterSpec("theta", (), ConstraintType.LOWER_UPPER, bounds=(0.0, 1.0)),
            ParameterSpec("mu", (2,), ConstraintType.UNCONSTRAINED),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        theta = params["theta"]
        mu = params["mu"]

        y = data["y"]
        lp1 = tfd.Normal(loc=mu[0], scale=1.0).log_prob(y)
        lp2 = tfd.Normal(loc=mu[1], scale=1.0).log_prob(y)
        mix = jnp.logaddexp(jnp.log(theta) + lp1, jnp.log1p(-theta) + lp2)

        lp = tfd.Uniform(low=0.0, high=1.0).log_prob(theta)
        lp = lp + jnp.sum(tfd.Normal(loc=0.0, scale=10.0).log_prob(mu))
        lp = lp + jnp.sum(mix)
        return lp
