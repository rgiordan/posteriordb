"""JAX/TFP translation of posterior_database/models/stan/M0_model.stan:

    data { array[M, T] int<lower=0, upper=1> y; }
    transformed data { s[i] = sum(y[i]); C = count of s[i] > 0; }
    parameters { real<lower=0, upper=1> omega; real<lower=0, upper=1> p; }
    model {
      for (i in 1:M) {
        if (s[i] > 0) {
          target += bernoulli_lpmf(1|omega) + binomial_lpmf(s[i]|T, p);
        } else {
          target += log_sum_exp(bernoulli_lpmf(1|omega) + binomial_lpmf(0|T, p),
                                bernoulli_lpmf(0|omega));
        }
      }
    }

A closure (`M0`) capture-recapture model: `s[i] = sum(y[i])` (each animal's
total detections) is a deterministic function of `data["y"]`, computed here
directly rather than at a separate "transformed data" step. Fully
vectorized over the `M` animals via `jnp.where` (`s > 0` selects which of
the two `target +=` branches applies to each animal).
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class M0Model(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        return [
            ParameterSpec("omega", (), ConstraintType.LOWER_UPPER, bounds=(0.0, 1.0)),
            ParameterSpec("p", (), ConstraintType.LOWER_UPPER, bounds=(0.0, 1.0)),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        omega = params["omega"]
        p = params["p"]

        y = jnp.asarray(data["y"], dtype=jnp.float64)  # (M, T)
        t = data["T"]
        s = jnp.sum(y, axis=1)  # (M,)

        log_bern1 = tfd.Bernoulli(probs=omega).log_prob(1)
        log_bern0 = tfd.Bernoulli(probs=omega).log_prob(0)
        log_binom_s = tfd.Binomial(total_count=float(t), probs=p).log_prob(s)
        log_binom_0 = tfd.Binomial(total_count=float(t), probs=p).log_prob(0.0)

        detected = log_bern1 + log_binom_s
        undetected = jnp.logaddexp(log_bern1 + log_binom_0, log_bern0)

        return jnp.sum(jnp.where(s > 0, detected, undetected))
