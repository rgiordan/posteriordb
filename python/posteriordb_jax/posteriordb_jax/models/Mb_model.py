"""JAX/TFP translation of posterior_database/models/stan/Mb_model.stan:

    data { array[M, T] int<lower=0, upper=1> y; }
    transformed data { s[i] = sum(y[i]); C = count of s[i] > 0; }
    parameters {
      real<lower=0, upper=1> omega; real<lower=0, upper=1> p; real<lower=0, upper=1> c;
    }
    transformed parameters {
      p_eff[i,1] = p;
      p_eff[i,j] = (1 - y[i,j-1]) * p + y[i,j-1] * c;  // j in 2..T
    }
    model {
      for (i in 1:M) {
        if (s[i] > 0) target += bernoulli_lpmf(1|omega) + bernoulli_lpmf(y[i]|p_eff[i]);
        else target += log_sum_exp(bernoulli_lpmf(1|omega) + bernoulli_lpmf(y[i]|p_eff[i]),
                                    bernoulli_lpmf(0|omega));
      }
    }

Same latent present/absent capture-recapture mixture as `M0_model`/
`Mt_model`, but the per-occasion detection probability `p_eff[i,j]` now
depends on the *behavioral response* to capture at the previous occasion:
`p` if not previously captured, `c` if it was ("trap happy"/"trap shy").
`p_eff` is built directly from the actual data `y` (like `Mt_model`'s `p`),
so when `s[i] == 0` (never detected) every `y[i,j-1] == 0` and `p_eff[i,:]`
already reduces to what the "undetected" branch's `bernoulli_lpmf(y[i] |
p_eff[i])` term needs -- no separate `s == 0` case, fully vectorized over
`M` via `jnp.where`, exactly as `Mt_model`'s docstring explains.
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class MbModel(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        return [
            ParameterSpec("omega", (), ConstraintType.LOWER_UPPER, bounds=(0.0, 1.0)),
            ParameterSpec("p", (), ConstraintType.LOWER_UPPER, bounds=(0.0, 1.0)),
            ParameterSpec("c", (), ConstraintType.LOWER_UPPER, bounds=(0.0, 1.0)),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        omega = params["omega"]
        p = params["p"]
        c = params["c"]

        y = jnp.asarray(data["y"], dtype=jnp.float64)  # (M, T)
        m = y.shape[0]
        s = jnp.sum(y, axis=1)  # (M,)

        p_eff_first = jnp.broadcast_to(p, (m, 1))
        p_eff_rest = (1.0 - y[:, :-1]) * p + y[:, :-1] * c  # (M, T-1)
        p_eff = jnp.concatenate([p_eff_first, p_eff_rest], axis=1)  # (M, T)

        per_animal = jnp.sum(tfd.Bernoulli(probs=p_eff).log_prob(y), axis=1)  # (M,)
        log_omega1 = tfd.Bernoulli(probs=omega).log_prob(1)
        log_omega0 = tfd.Bernoulli(probs=omega).log_prob(0)

        detected = log_omega1 + per_animal
        undetected = jnp.logaddexp(detected, log_omega0)

        return jnp.sum(jnp.where(s > 0, detected, undetected))
