"""JAX/TFP translation of posterior_database/models/stan/arma11.stan:

    parameters { real mu; real phi; real theta; real<lower=0> sigma; }
    model {
      mu ~ normal(0, 10); phi ~ normal(0, 2); theta ~ normal(0, 2); sigma ~ cauchy(0, 2.5);
      nu[1] = mu + phi * mu;  // assume err[0] == 0
      err[1] = y[1] - nu[1];
      for (t in 2:T) {
        nu[t] = mu + phi * y[t-1] + theta * err[t-1];
        err[t] = y[t] - nu[t];
      }
      err ~ normal(0, sigma);
    }

`err[t]` depends on `err[t-1]`, a genuine sequential recursion; implemented
with `jax.lax.scan` (differentiable, and doesn't unroll a length-T Python
loop into the trace).
"""

from __future__ import annotations

from typing import Dict, List

import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class Arma11(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        return [
            ParameterSpec("mu", (), ConstraintType.UNCONSTRAINED),
            ParameterSpec("phi", (), ConstraintType.UNCONSTRAINED),
            ParameterSpec("theta", (), ConstraintType.UNCONSTRAINED),
            ParameterSpec("sigma", (), ConstraintType.LOWER, bounds=(0.0, None)),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        mu = params["mu"]
        phi = params["phi"]
        theta = params["theta"]
        sigma = params["sigma"]

        y = data["y"]
        err0 = y[0] - (mu + phi * mu)

        def step(err_prev, y_t_ytm1):
            y_t, y_tm1 = y_t_ytm1
            nu_t = mu + phi * y_tm1 + theta * err_prev
            err_t = y_t - nu_t
            return err_t, err_t

        _, err_rest = jax.lax.scan(step, err0, (y[1:], y[:-1]))
        err = jnp.concatenate([err0[None], err_rest])

        lp = tfd.Normal(loc=0.0, scale=10.0).log_prob(mu)
        lp = lp + tfd.Normal(loc=0.0, scale=2.0).log_prob(phi)
        lp = lp + tfd.Normal(loc=0.0, scale=2.0).log_prob(theta)
        lp = lp + tfd.Cauchy(loc=0.0, scale=2.5).log_prob(sigma)
        lp = lp + jnp.sum(tfd.Normal(loc=0.0, scale=sigma).log_prob(err))
        return lp
