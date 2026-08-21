"""JAX/TFP translation of posterior_database/models/stan/dugongs_model.stan:

    parameters {
      real alpha; real beta;
      real<lower=.5, upper=1> lambda;
      real<lower=0> tau;
    }
    transformed parameters { real sigma = 1 / sqrt(tau); real U3 = logit(lambda); }
    model {
      m[i] = alpha - beta * pow(lambda, x[i]);
      Y ~ normal(m, sigma);
      alpha ~ normal(0.0, 1000);
      beta ~ normal(0.0, 1000);
      lambda ~ uniform(.5, 1);
      tau ~ gamma(.0001, .0001);
    }

`U3` (derived, unused elsewhere) is omitted. `lambda ~ uniform(.5, 1)` has a
nonzero constant lpdf (`-log(0.5)`, unlike a `uniform(0,1)` prior) so it's
included explicitly for an exact log-density value match. Stan's
`gamma(alpha, beta)` is shape/rate, matching `tfd.Gamma`'s
`concentration`/`rate` parameterization directly.
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class DugongsModel(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        return [
            ParameterSpec("alpha", (), ConstraintType.UNCONSTRAINED),
            ParameterSpec("beta", (), ConstraintType.UNCONSTRAINED),
            ParameterSpec("lambda", (), ConstraintType.LOWER_UPPER, bounds=(0.5, 1.0)),
            ParameterSpec("tau", (), ConstraintType.LOWER, bounds=(0.0, None)),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        alpha = params["alpha"]
        beta = params["beta"]
        lam = params["lambda"]
        tau = params["tau"]

        sigma = 1.0 / jnp.sqrt(tau)
        x = data["x"]
        m = alpha - beta * jnp.power(lam, x)

        lp = jnp.sum(tfd.Normal(loc=m, scale=sigma).log_prob(data["Y"]))
        lp = lp + tfd.Normal(loc=0.0, scale=1000.0).log_prob(alpha)
        lp = lp + tfd.Normal(loc=0.0, scale=1000.0).log_prob(beta)
        lp = lp + tfd.Uniform(low=0.5, high=1.0).log_prob(lam)
        lp = lp + tfd.Gamma(concentration=1e-4, rate=1e-4).log_prob(tau)
        return lp
