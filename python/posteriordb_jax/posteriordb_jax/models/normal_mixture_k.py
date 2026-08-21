"""JAX/TFP translation of posterior_database/models/stan/normal_mixture_k.stan:

    parameters {
      simplex[K] theta;
      array[K] real mu;
      array[K] real<lower=0, upper=10> sigma;
    }
    model {
      mu ~ normal(0, 10);
      for (n in 1 : N) {
        for (k in 1 : K) { ps[k] = log(theta[k]) + normal_lpdf(y[n] | mu[k], sigma[k]); }
        target += log_sum_exp(ps);
      }
    }
"""

from __future__ import annotations

from typing import Dict, List

import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class NormalMixtureK(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        k = int(data["K"])
        return [
            ParameterSpec("theta", (k,), ConstraintType.SIMPLEX),
            ParameterSpec("mu", (k,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("sigma", (k,), ConstraintType.LOWER_UPPER, bounds=(0.0, 10.0)),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        theta = params["theta"]
        mu = params["mu"]
        sigma = params["sigma"]

        y = data["y"]  # (N,)
        # ps[k, n] = log(theta[k]) + Normal(mu[k], sigma[k]).log_prob(y[n])
        ps = jnp.log(theta)[:, None] + tfd.Normal(loc=mu[:, None], scale=sigma[:, None]).log_prob(y[None, :])

        lp = jnp.sum(tfd.Normal(loc=0.0, scale=10.0).log_prob(mu))
        lp = lp + jnp.sum(jax.scipy.special.logsumexp(ps, axis=0))
        return lp
