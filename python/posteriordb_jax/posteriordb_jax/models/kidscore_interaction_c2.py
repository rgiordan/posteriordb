"""JAX/TFP translation of posterior_database/models/stan/kidscore_interaction_c2.stan:

    transformed data {
      vector[N] c2_mom_hs = mom_hs - 0.5;
      vector[N] c2_mom_iq = mom_iq - 100;
      vector[N] inter = c2_mom_hs .* c2_mom_iq;
    }
    parameters { vector[4] beta; real<lower=0> sigma; }
    model {
      kid_score ~ normal(beta[1] + beta[2] * c2_mom_hs + beta[3] * c2_mom_iq
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


class KidscoreInteractionC2(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        return [
            ParameterSpec("beta", (4,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("sigma", (), ConstraintType.LOWER, bounds=(0.0, None)),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        beta = params["beta"]
        sigma = params["sigma"]
        c2_mom_hs = data["mom_hs"] - 0.5
        c2_mom_iq = data["mom_iq"] - 100.0
        inter = c2_mom_hs * c2_mom_iq
        mu = beta[0] + beta[1] * c2_mom_hs + beta[2] * c2_mom_iq + beta[3] * inter
        return jnp.sum(tfd.Normal(loc=mu, scale=sigma).log_prob(data["kid_score"]))
