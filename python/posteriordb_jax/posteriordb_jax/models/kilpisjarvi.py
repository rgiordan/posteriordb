"""JAX/TFP translation of posterior_database/models/stan/kilpisjarvi.stan:

    data { ... real pmualpha; real psalpha; real pmubeta; real psbeta; }
    parameters { real alpha; real beta; real<lower=0> sigma; }
    model {
      alpha ~ normal(pmualpha, psalpha);
      beta ~ normal(pmubeta, psbeta);
      y ~ normal(alpha + beta * x, sigma);
    }

The prior hyperparameters (`pmualpha`, `psalpha`, `pmubeta`, `psbeta`) are
themselves Stan `data`, not fixed constants.
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class Kilpisjarvi(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        return [
            ParameterSpec("alpha", (), ConstraintType.UNCONSTRAINED),
            ParameterSpec("beta", (), ConstraintType.UNCONSTRAINED),
            ParameterSpec("sigma", (), ConstraintType.LOWER, bounds=(0.0, None)),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        alpha = params["alpha"]
        beta = params["beta"]
        sigma = params["sigma"]

        mu = alpha + beta * data["x"]
        lp = tfd.Normal(loc=data["pmualpha"], scale=data["psalpha"]).log_prob(alpha)
        lp = lp + tfd.Normal(loc=data["pmubeta"], scale=data["psbeta"]).log_prob(beta)
        lp = lp + jnp.sum(tfd.Normal(loc=mu, scale=sigma).log_prob(data["y"]))
        return lp
