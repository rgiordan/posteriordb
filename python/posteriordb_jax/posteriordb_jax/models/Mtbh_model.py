"""JAX/TFP translation of posterior_database/models/stan/Mtbh_model.stan:

    data { array[M, T] int<lower=0, upper=1> y; }
    parameters {
      real<lower=0, upper=1> omega; vector<lower=0, upper=1>[T] mean_p;
      real gamma; real<lower=0, upper=3> sigma; vector[M] eps_raw;
    }
    transformed parameters {
      eps = sigma * eps_raw;                          // (M,)
      alpha = logit(mean_p);                           // (T,)
      logit_p[:, 1] = alpha[1] + eps;                   // no trap-response term on occasion 1
      logit_p[i, j] = alpha[j] + eps[i] + gamma * y[i, j-1];  // j in 2..T
    }
    model {
      gamma ~ normal(0, 10);
      eps_raw ~ normal(0, 1);
      for (i in 1:M) {
        if (s[i] > 0) target += bernoulli_lpmf(1|omega) + bernoulli_logit_lpmf(y[i]|logit_p[i]);
        else target += log_sum_exp(bernoulli_lpmf(1|omega) + bernoulli_logit_lpmf(y[i]|logit_p[i]),
                                    bernoulli_lpmf(0|omega));
      }
    }

`Mth_model` plus a trap-response term `gamma * y[i, j-1]` (behavioral
response to capture at the *previous* occasion, like `Mb_model`'s `c`, but
additive on the logit scale and shared across occasions rather than a
separate probability). Since `logit_p` is built from the actual data `y`
(including the `gamma * y[i, j-1]` term, which is `0` whenever `s[i] == 0`
makes the whole row all-zero), the same no-separate-`s==0`-case
simplification as `Mt_model`/`Mb_model`/`Mth_model` still applies.
"""

from __future__ import annotations

from typing import Dict, List

import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class MtbhModel(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        m = int(data["M"])
        t = int(data["T"])
        return [
            ParameterSpec("omega", (), ConstraintType.LOWER_UPPER, bounds=(0.0, 1.0)),
            ParameterSpec("mean_p", (t,), ConstraintType.LOWER_UPPER, bounds=(0.0, 1.0)),
            ParameterSpec("gamma", (), ConstraintType.UNCONSTRAINED),
            ParameterSpec("sigma", (), ConstraintType.LOWER_UPPER, bounds=(0.0, 3.0)),
            ParameterSpec("eps_raw", (m,), ConstraintType.UNCONSTRAINED),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        omega = params["omega"]
        mean_p = params["mean_p"]
        gamma = params["gamma"]
        sigma = params["sigma"]
        eps_raw = params["eps_raw"]

        y = jnp.asarray(data["y"], dtype=jnp.float64)  # (M, T)
        m = y.shape[0]
        s = jnp.sum(y, axis=1)  # (M,)

        eps = sigma * eps_raw  # (M,)
        alpha = jax.scipy.special.logit(mean_p)  # (T,)

        y_prev = jnp.concatenate([jnp.zeros((m, 1), dtype=y.dtype), y[:, :-1]], axis=1)  # (M, T)
        logit_p = alpha[None, :] + eps[:, None] + gamma * y_prev  # (M, T)

        per_animal = jnp.sum(tfd.Bernoulli(logits=logit_p).log_prob(y), axis=1)  # (M,)
        log_omega1 = tfd.Bernoulli(probs=omega).log_prob(1)
        log_omega0 = tfd.Bernoulli(probs=omega).log_prob(0)

        detected = log_omega1 + per_animal
        undetected = jnp.logaddexp(detected, log_omega0)

        lp = jnp.sum(jnp.where(s > 0, detected, undetected))
        lp = lp + tfd.Normal(loc=0.0, scale=10.0).log_prob(gamma)
        lp = lp + jnp.sum(tfd.Normal(loc=0.0, scale=1.0).log_prob(eps_raw))
        return lp
