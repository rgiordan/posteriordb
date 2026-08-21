"""JAX/TFP translation of
posterior_database/models/stan/eight_schools_centered.stan:

    data {
      int<lower=0> J;
      array[J] real y;
      array[J] real<lower=0> sigma;
    }
    parameters {
      array[J] real theta;
      real mu;
      real<lower=0> tau;
    }
    model {
      tau ~ cauchy(0, 5);
      theta ~ normal(mu, tau);
      y ~ normal(theta, sigma);
      mu ~ normal(0, 5);
    }

Unlike `eight_schools_noncentered`, `theta` here is Stan's own raw sampled
parameter, so posteriordb's reference draws (keyed by `theta`, `mu`, `tau`)
match this model's parameter names directly -- no adapter needed.
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class EightSchoolsCentered(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        j = int(data["J"])
        return [
            ParameterSpec("theta", (j,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("mu", (), ConstraintType.UNCONSTRAINED),
            ParameterSpec("tau", (), ConstraintType.LOWER, bounds=(0.0, None)),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        theta = params["theta"]
        mu = params["mu"]
        tau = params["tau"]

        y = data["y"]
        sigma = data["sigma"]

        lp = tfd.Cauchy(loc=0.0, scale=5.0).log_prob(tau)
        lp = lp + jnp.sum(tfd.Normal(loc=mu, scale=tau).log_prob(theta))
        lp = lp + jnp.sum(tfd.Normal(loc=theta, scale=sigma).log_prob(y))
        lp = lp + tfd.Normal(loc=0.0, scale=5.0).log_prob(mu)
        return lp
