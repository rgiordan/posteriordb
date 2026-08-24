"""JAX/TFP translation of posterior_database/models/stan/Mth_model.stan:

    data { array[M, T] int<lower=0, upper=1> y; }
    parameters {
      real<lower=0, upper=1> omega; array[T] real<lower=0, upper=1> mean_p;
      real<lower=0, upper=5> sigma; vector[M] eps_raw;
    }
    transformed parameters {
      eps = sigma * eps_raw;                 // (M,) individual random effect
      mean_lp = logit(mean_p);               // (T,)
      logit_p[:, j] = mean_lp[j] + eps;       // (M, T)
    }
    model {
      eps_raw ~ normal(0, 1);
      for (i in 1:M) {
        if (s[i] > 0) target += bernoulli_lpmf(1|omega) + bernoulli_logit_lpmf(y[i]|logit_p[i]);
        else target += log_sum_exp(bernoulli_lpmf(1|omega) + bernoulli_logit_lpmf(y[i]|logit_p[i]),
                                    bernoulli_lpmf(0|omega));
      }
    }

Combines `Mt_model`'s per-occasion detection probability `mean_p[t]` with
`Mh_model`'s individual random effect `eps[i]` (additive on the logit
scale). `logit_p` depends only on parameters (not `y`), so, exactly as in
`Mt_model`/`Mb_model`/`Mh_model`, `bernoulli_logit_lpmf(y[i] | logit_p[i])`
evaluated at the actual (all-zero, when `s[i] == 0`) data row already equals
what the "undetected" branch needs -- no separate `s == 0` case.
"""

from __future__ import annotations

from typing import Dict, List

import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class MthModel(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        m = int(data["M"])
        t = int(data["T"])
        return [
            ParameterSpec("omega", (), ConstraintType.LOWER_UPPER, bounds=(0.0, 1.0)),
            ParameterSpec("mean_p", (t,), ConstraintType.LOWER_UPPER, bounds=(0.0, 1.0)),
            ParameterSpec("sigma", (), ConstraintType.LOWER_UPPER, bounds=(0.0, 5.0)),
            ParameterSpec("eps_raw", (m,), ConstraintType.UNCONSTRAINED),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        omega = params["omega"]
        mean_p = params["mean_p"]
        sigma = params["sigma"]
        eps_raw = params["eps_raw"]

        y = jnp.asarray(data["y"], dtype=jnp.float64)  # (M, T)
        s = jnp.sum(y, axis=1)  # (M,)

        eps = sigma * eps_raw  # (M,)
        mean_lp = jax.scipy.special.logit(mean_p)  # (T,)
        logit_p = mean_lp[None, :] + eps[:, None]  # (M, T)

        per_animal = jnp.sum(tfd.Bernoulli(logits=logit_p).log_prob(y), axis=1)  # (M,)
        log_omega1 = tfd.Bernoulli(probs=omega).log_prob(1)
        log_omega0 = tfd.Bernoulli(probs=omega).log_prob(0)

        detected = log_omega1 + per_animal
        undetected = jnp.logaddexp(detected, log_omega0)

        lp = jnp.sum(jnp.where(s > 0, detected, undetected))
        lp = lp + jnp.sum(tfd.Normal(loc=0.0, scale=1.0).log_prob(eps_raw))
        return lp
