"""JAX/TFP translation of posterior_database/models/stan/kidscore_mom_work.stan:

    transformed data {
      vector[N] work2; vector[N] work3; vector[N] work4;
      for (i) { work2[i] = mom_work[i]==2; work3[i] = mom_work[i]==3; work4[i] = mom_work[i]==4; }
    }
    parameters { vector[4] beta; real<lower=0> sigma; }
    model {
      kid_score ~ normal(beta[1] + beta[2] * work2 + beta[3] * work3
                         + beta[4] * work4, sigma);
    }
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class KidscoreMomWork(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        return [
            ParameterSpec("beta", (4,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("sigma", (), ConstraintType.LOWER, bounds=(0.0, None)),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        beta = params["beta"]
        sigma = params["sigma"]
        mom_work = jnp.asarray(data["mom_work"])
        work2 = (mom_work == 2).astype(jnp.float64)
        work3 = (mom_work == 3).astype(jnp.float64)
        work4 = (mom_work == 4).astype(jnp.float64)
        mu = beta[0] + beta[1] * work2 + beta[2] * work3 + beta[3] * work4
        return jnp.sum(tfd.Normal(loc=mu, scale=sigma).log_prob(data["kid_score"]))
