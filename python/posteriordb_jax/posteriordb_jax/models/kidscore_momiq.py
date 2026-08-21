"""JAX/TFP translation of posterior_database/models/stan/kidscore_momiq.stan:

    parameters { vector[2] beta; real<lower=0> sigma; }
    model {
      sigma ~ cauchy(0, 2.5);
      kid_score ~ normal(beta[1] + beta[2] * mom_iq, sigma);
    }
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class KidscoreMomiq(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        return [
            ParameterSpec("beta", (2,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("sigma", (), ConstraintType.LOWER, bounds=(0.0, None)),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        beta = params["beta"]
        sigma = params["sigma"]
        mu = beta[0] + beta[1] * data["mom_iq"]
        lp = tfd.Cauchy(loc=0.0, scale=2.5).log_prob(sigma)
        lp = lp + jnp.sum(tfd.Normal(loc=mu, scale=sigma).log_prob(data["kid_score"]))
        return lp
