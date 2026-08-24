"""JAX/TFP translation of posterior_database/models/stan/Mh_model.stan:

    data { array[M] int<lower=0, upper=T> y; }  // total detections per animal
    transformed data { C = count of y[i] > 0. }
    parameters {
      real<lower=0, upper=1> omega; real<lower=0, upper=1> mean_p;
      real<lower=0, upper=5> sigma; vector[M] eps_raw;
    }
    transformed parameters { eps = logit(mean_p) + sigma * eps_raw; }
    model {
      eps_raw ~ normal(0, 1);
      for (i in 1:M) {
        if (y[i] > 0) target += bernoulli_lpmf(1|omega) + binomial_logit_lpmf(y[i]|T, eps[i]);
        else target += log_sum_exp(bernoulli_lpmf(1|omega) + binomial_logit_lpmf(0|T, eps[i]),
                                    bernoulli_lpmf(0|omega));
      }
    }

Same latent present/absent mixture family as `M0_model`/`Mt_model`/
`Mb_model`, but with an individual random effect `eps[i]` (heterogeneous
per-animal detection probability, on the logit scale) instead of a shared
or occasion-varying one, and the per-animal likelihood is `binomial_logit`
over the *total* detection count `y[i]` (not per-occasion binary). `eps`
depends only on parameters (not on `y`), so `binomial_logit_lpmf(y[i] | T,
eps[i])` already equals what the `y[i] == 0` branch needs when `y[i] == 0`
-- same no-separate-case simplification as `Mt_model`/`Mb_model`.
"""

from __future__ import annotations

from typing import Dict, List

import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class MhModel(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        m = int(data["M"])
        return [
            ParameterSpec("omega", (), ConstraintType.LOWER_UPPER, bounds=(0.0, 1.0)),
            ParameterSpec("mean_p", (), ConstraintType.LOWER_UPPER, bounds=(0.0, 1.0)),
            ParameterSpec("sigma", (), ConstraintType.LOWER_UPPER, bounds=(0.0, 5.0)),
            ParameterSpec("eps_raw", (m,), ConstraintType.UNCONSTRAINED),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        omega = params["omega"]
        mean_p = params["mean_p"]
        sigma = params["sigma"]
        eps_raw = params["eps_raw"]

        y = jnp.asarray(data["y"], dtype=jnp.float64)  # (M,)
        t = float(data["T"])

        eps = jax.scipy.special.logit(mean_p) + sigma * eps_raw  # (M,)
        binom_lp = tfd.Binomial(total_count=t, logits=eps).log_prob(y)  # (M,)

        log_omega1 = tfd.Bernoulli(probs=omega).log_prob(1)
        log_omega0 = tfd.Bernoulli(probs=omega).log_prob(0)

        detected = log_omega1 + binom_lp
        undetected = jnp.logaddexp(detected, log_omega0)

        lp = jnp.sum(jnp.where(y > 0, detected, undetected))
        lp = lp + jnp.sum(tfd.Normal(loc=0.0, scale=1.0).log_prob(eps_raw))
        return lp
