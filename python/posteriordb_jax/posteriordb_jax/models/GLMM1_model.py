"""JAX/TFP translation of posterior_database/models/stan/GLMM1_model.stan:

    parameters { vector[nsite] alpha; real mu_alpha; real<lower=0, upper=5> sd_alpha; }
    transformed parameters { log_lambda = rep_matrix(alpha', nyear); }
    model {
      alpha ~ normal(mu_alpha, sd_alpha);
      mu_alpha ~ normal(0, 10);
      // sd_alpha ~ uniform(0, 5);  // Implicitly defined (bound only, no term)
      obs[i] ~ poisson_log(log_lambda[obsyear[i], obssite[i]]);
    }

`log_lambda[year, site] = alpha[site]` regardless of `year` (a
`rep_matrix` broadcast), so the observed likelihood only ever needs
`alpha[obssite[i]]` -- `obsyear` is unused by the model as written. (The
`generated quantities` missing-data imputation is RNG-only and omitted.)
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class GLMM1Model(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        nsite = int(data["nsite"])
        return [
            ParameterSpec("alpha", (nsite,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("mu_alpha", (), ConstraintType.UNCONSTRAINED),
            ParameterSpec("sd_alpha", (), ConstraintType.LOWER_UPPER, bounds=(0.0, 5.0)),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        alpha = params["alpha"]
        mu_alpha = params["mu_alpha"]
        sd_alpha = params["sd_alpha"]

        obssite = jnp.asarray(data["obssite"]) - 1
        log_lambda_obs = alpha[obssite]
        obs = jnp.asarray(data["obs"], dtype=jnp.float64)

        lp = jnp.sum(tfd.Normal(loc=mu_alpha, scale=sd_alpha).log_prob(alpha))
        lp = lp + tfd.Normal(loc=0.0, scale=10.0).log_prob(mu_alpha)
        lp = lp + jnp.sum(tfd.Poisson(log_rate=log_lambda_obs).log_prob(obs))
        return lp
