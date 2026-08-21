"""JAX/TFP translation of posterior_database/models/stan/nes.stan:

    transformed data {
      age30_44[n] = age_discrete[n] == 2; age45_64[n] = age_discrete[n] == 3; age65up[n] = age_discrete[n] == 4;
    }
    parameters { vector[9] beta; real<lower=0> sigma; }
    model {
      partyid7 ~ normal(beta[1] + beta[2]*real_ideo + beta[3]*race_adj
                        + beta[4]*age30_44 + beta[5]*age45_64 + beta[6]*age65up
                        + beta[7]*educ1 + beta[8]*gender + beta[9]*income, sigma);
    }
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class Nes(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        return [
            ParameterSpec("beta", (9,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("sigma", (), ConstraintType.LOWER, bounds=(0.0, None)),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        beta = params["beta"]
        sigma = params["sigma"]

        age_discrete = jnp.asarray(data["age_discrete"])
        age30_44 = (age_discrete == 2).astype(jnp.float64)
        age45_64 = (age_discrete == 3).astype(jnp.float64)
        age65up = (age_discrete == 4).astype(jnp.float64)

        mu = (
            beta[0]
            + beta[1] * data["real_ideo"]
            + beta[2] * data["race_adj"]
            + beta[3] * age30_44
            + beta[4] * age45_64
            + beta[5] * age65up
            + beta[6] * data["educ1"]
            + beta[7] * data["gender"]
            + beta[8] * data["income"]
        )
        return jnp.sum(tfd.Normal(loc=mu, scale=sigma).log_prob(data["partyid7"]))
