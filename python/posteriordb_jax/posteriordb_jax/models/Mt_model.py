"""JAX/TFP translation of posterior_database/models/stan/Mt_model.stan:

    data { array[M, T] int<lower=0, upper=1> y; }
    parameters { real<lower=0, upper=1> omega; vector<lower=0, upper=1>[T] p; }
    model {
      for (i in 1:M) {
        if (s[i] > 0) {
          target += bernoulli_lpmf(1|omega) + bernoulli_lpmf(y[i]|p);
        } else {
          target += log_sum_exp(bernoulli_lpmf(1|omega) + bernoulli_lpmf(y[i]|p),
                                bernoulli_lpmf(0|omega));
        }
      }
    }

Like `M0_model` but with a per-occasion detection probability `p[t]`
(rather than one shared `p`), so the animal-level likelihood is
`sum_t bernoulli_lpmf(y[i,t] | p[t])`, not a `binomial`. When `s[i] == 0`,
`y[i,:]` is exactly all-zero, so this same per-animal sum already equals
what the "undetected" branch's first term needs -- no separate `s == 0`
case is needed, fully vectorized over `M` via `jnp.where`.
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class MtModel(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        t = int(data["T"])
        return [
            ParameterSpec("omega", (), ConstraintType.LOWER_UPPER, bounds=(0.0, 1.0)),
            ParameterSpec("p", (t,), ConstraintType.LOWER_UPPER, bounds=(0.0, 1.0)),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        omega = params["omega"]
        p = params["p"]

        y = jnp.asarray(data["y"], dtype=jnp.float64)  # (M, T)
        s = jnp.sum(y, axis=1)  # (M,)

        per_animal_bern = jnp.sum(tfd.Bernoulli(probs=p[None, :]).log_prob(y), axis=1)  # (M,)
        log_omega1 = tfd.Bernoulli(probs=omega).log_prob(1)
        log_omega0 = tfd.Bernoulli(probs=omega).log_prob(0)

        detected = log_omega1 + per_animal_bern
        undetected = jnp.logaddexp(log_omega1 + per_animal_bern, log_omega0)

        return jnp.sum(jnp.where(s > 0, detected, undetected))
