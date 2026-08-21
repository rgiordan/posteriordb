"""JAX/TFP translation of posterior_database/models/stan/kidscore_interaction_z.stan:

    transformed data {
      vector[N] z_mom_hs = (mom_hs - mean(mom_hs)) / (2 * sd(mom_hs));
      vector[N] z_mom_iq = (mom_iq - mean(mom_iq)) / (2 * sd(mom_iq));
      vector[N] inter = z_mom_hs .* z_mom_iq;
    }
    parameters { vector[4] beta; real<lower=0> sigma; }
    model {
      kid_score ~ normal(beta[1] + beta[2] * z_mom_hs + beta[3] * z_mom_iq
                         + beta[4] * inter, sigma);
    }

Stan's `sd()` is the sample standard deviation (`ddof=1`).
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class KidscoreInteractionZ(StatisticalModel):
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
        z_mom_hs = (mom_hs - jnp.mean(mom_hs)) / (2.0 * jnp.std(mom_hs, ddof=1))
        z_mom_iq = (mom_iq - jnp.mean(mom_iq)) / (2.0 * jnp.std(mom_iq, ddof=1))
        inter = z_mom_hs * z_mom_iq
        mu = beta[0] + beta[1] * z_mom_hs + beta[2] * z_mom_iq + beta[3] * inter
        return jnp.sum(tfd.Normal(loc=mu, scale=sigma).log_prob(data["kid_score"]))
