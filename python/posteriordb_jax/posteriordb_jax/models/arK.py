"""JAX/TFP translation of posterior_database/models/stan/arK.stan:

    data {
      int<lower=0> K;
      int<lower=0> T;
      array[T] real y;
    }
    parameters {
      real alpha;
      array[K] real beta;
      real<lower=0> sigma;
    }
    model {
      alpha ~ normal(0, 10);
      beta ~ normal(0, 10);
      sigma ~ cauchy(0, 2.5);

      for (t in (K + 1) : T) {
        real mu;
        mu = alpha;
        for (k in 1 : K) {
          mu = mu + beta[k] * y[t - k];
        }
        y[t] ~ normal(mu, sigma);
      }
    }

The `t`-loop is vectorized: with 0-indexed `y` of length `T`, Stan's
`mu[t] = alpha + sum_k beta[k] * y[t - k]` for `t` in `[K+1, T]` (1-indexed)
becomes, for 0-indexed `it` in `[K, T-1]`,
`mu[it] = alpha + sum_{k=1}^K beta[k-1] * y[it - k]` -- a sum of `K` shifted
slices of `y`, one per lag (`K` is small and static per dataset, so a Python
loop over lags, not timesteps, is fine to trace).
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class ArK(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        k = int(data["K"])
        return [
            ParameterSpec("alpha", (), ConstraintType.UNCONSTRAINED),
            ParameterSpec("beta", (k,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("sigma", (), ConstraintType.LOWER, bounds=(0.0, None)),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        alpha = params["alpha"]
        beta = params["beta"]
        sigma = params["sigma"]

        y = data["y"]
        k = int(data["K"])
        t = int(data["T"])

        mu = alpha
        for lag in range(1, k + 1):
            mu = mu + beta[lag - 1] * y[k - lag : t - lag]
        y_tail = y[k:t]

        lp = jnp.sum(tfd.Normal(loc=0.0, scale=10.0).log_prob(alpha))
        lp = lp + jnp.sum(tfd.Normal(loc=0.0, scale=10.0).log_prob(beta))
        lp = lp + tfd.Cauchy(loc=0.0, scale=2.5).log_prob(sigma)
        lp = lp + jnp.sum(tfd.Normal(loc=mu, scale=sigma).log_prob(y_tail))
        return lp
