"""JAX/TFP translation of posterior_database/models/stan/kidscore_interaction.stan:

    transformed data { vector[N] inter = mom_hs .* mom_iq; }
    parameters { vector[4] beta; real<lower=0> sigma; }
    model {
      sigma ~ cauchy(0, 2.5);
      kid_score ~ normal(beta[1] + beta[2] * mom_hs + beta[3] * mom_iq
                         + beta[4] * inter, sigma);
    }
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class KidscoreInteraction(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        return [
            ParameterSpec("beta", (4,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("sigma", (), ConstraintType.LOWER, bounds=(0.0, None)),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        beta = params["beta"]
        sigma = params["sigma"]
        mom_hs = data["mom_hs"]
        mom_iq = data["mom_iq"]
        inter = mom_hs * mom_iq
        mu = beta[0] + beta[1] * mom_hs + beta[2] * mom_iq + beta[3] * inter
        lp = tfd.Cauchy(loc=0.0, scale=2.5).log_prob(sigma)
        lp = lp + jnp.sum(tfd.Normal(loc=mu, scale=sigma).log_prob(data["kid_score"]))
        return lp
