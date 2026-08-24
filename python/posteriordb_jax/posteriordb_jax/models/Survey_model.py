"""JAX/TFP translation of posterior_database/models/stan/Survey_model.stan:

    data { int<lower=0> nmax; int<lower=0> m; array[m] int<lower=0, upper=nmax> k; }
    transformed data { nmin = max(k); }
    parameters { real<lower=0, upper=1> theta; }
    transformed parameters {
      for (n in 1:nmax) {
        lp_parts[n] = log(1.0/nmax) + (n < nmin ? negative_infinity() : binomial_lpmf(k | n, theta));
      }
    }
    model { target += log_sum_exp(lp_parts); }

A discrete-latent-variable ("number of surveys sent", `n`) marginalization:
the model log density is `log_sum_exp` over a uniform prior on `n in
1..nmax` combined with `binomial_lpmf(k | n, theta)` summed over the `m`
observed return counts, but only for `n >= nmin = max(k)` (smaller `n` can't
have produced the largest observed return count, so Stan hard-codes those
`lp_parts` entries to `-inf` rather than ever calling `binomial_lpmf` with
an invalid `n < k` domain).

The `n < nmin` branch is masked on the *input* to `Binomial`, not just the
output: `total_count` is clamped to `nmin` (always >= every `k_m`) before
computing `log_prob`, and only the *resulting* (always-valid) log-probability
is then replaced by `-inf` via `jnp.where` for those `n`. Masking only the
output (computing `Binomial(total_count=n, ...).log_prob(k)` directly for
`n < k_m`, then discarding the result) would let an invalid-domain zero
probability's `-inf`/`nan` gradient leak into the total via
`jnp.where`'s untaken-branch autodiff (both branches are still
differentiated), corrupting `grad(theta)` even though the corresponding
forward value is masked away.
"""

from __future__ import annotations

from typing import Dict, List

import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class SurveyModel(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        return [ParameterSpec("theta", (), ConstraintType.LOWER_UPPER, bounds=(0.0, 1.0))]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        theta = params["theta"]

        nmax = int(data["nmax"])
        k = jnp.asarray(data["k"], dtype=jnp.float64)  # (m,)
        nmin = jnp.max(k)

        n = jnp.arange(1, nmax + 1, dtype=jnp.float64)  # (nmax,)
        valid = n >= nmin
        n_safe = jnp.where(valid, n, nmin)  # always >= every k_m: safe Binomial domain

        binom_lp = tfd.Binomial(total_count=n_safe[:, None], probs=theta).log_prob(k[None, :])  # (nmax, m)
        binom_sum = jnp.sum(binom_lp, axis=1)  # (nmax,)

        lp_parts = jnp.log(1.0 / nmax) + jnp.where(valid, binom_sum, -jnp.inf)
        return jax.scipy.special.logsumexp(lp_parts)
