"""JAX/TFP translation of
posterior_database/models/stan/dogs_nonhierarchical.stan: a non-centered
hierarchical logistic ("avoidance learning") model with a free
`cholesky_factor_corr[2]` parameter and a matrix-shaped free parameter `z`.

    parameters {
      vector[2] mu_logit_ab;
      vector<lower=0>[2] sigma_logit_ab;
      cholesky_factor_corr[2] L_logit_ab;
      matrix[J, 2] z;
    }
    transformed parameters {
      matrix[J, 2] logit_ab = rep_vector(1, J) * mu_logit_ab'
                              + z * diag_pre_multiply(sigma_logit_ab, L_logit_ab);
      vector[J] a = inv_logit(logit_ab[:, 1]);
      vector[J] b = inv_logit(logit_ab[:, 2]);
    }
    model {
      for (j, t) y[j,t] ~ bernoulli(a[j]^prev_shock[j,t] * b[j]^prev_avoid[j,t]);
      mu_logit_ab ~ logistic(0, 1);
      sigma_logit_ab ~ normal(0, 1);
      L_logit_ab ~ lkj_corr_cholesky(2);
      to_vector(z) ~ normal(0, 1);
    }

`prev_shock`/`prev_avoid` (Stan's `transformed data`) are recomputed here
from `data["y"]` each call rather than cached -- cheap, and keeps `data` a
plain pytree of the original Stan `data` block fields.

Note: this posterior has no stored posteriordb reference draws
(`reference_posterior_name: null` in `posteriors/dogs-dogs_nonhierarchical.json`),
so its test uses synthetic draws only (the testing framework's default).
"""

from __future__ import annotations

from typing import Dict, List

import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


def _prev_shock_avoid(y: jnp.ndarray):
    """y: (J, T) binary. Returns prev_shock, prev_avoid: (J, T), matching
    Stan's transformed-data block exactly (prev_*[:, 0] = 0, then a
    cumulative count of prior shocks/avoidances)."""
    j = y.shape[0]
    zeros_col = jnp.zeros((j, 1), dtype=y.dtype)
    prev_shock = jnp.concatenate([zeros_col, jnp.cumsum(y[:, :-1], axis=1)], axis=1)
    prev_avoid = jnp.concatenate([zeros_col, jnp.cumsum(1 - y[:, :-1], axis=1)], axis=1)
    return prev_shock, prev_avoid


class DogsNonhierarchical(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        j = int(data["n_dogs"])
        return [
            ParameterSpec("mu_logit_ab", (2,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("sigma_logit_ab", (2,), ConstraintType.LOWER, bounds=(0.0, None)),
            ParameterSpec("L_logit_ab", (2, 2), ConstraintType.CHOLESKY_FACTOR_CORR),
            ParameterSpec("z", (j, 2), ConstraintType.UNCONSTRAINED),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        mu_logit_ab = params["mu_logit_ab"]
        sigma_logit_ab = params["sigma_logit_ab"]
        l_logit_ab = params["L_logit_ab"]
        z = params["z"]

        y = jnp.asarray(data["y"], dtype=jnp.float64)
        prev_shock, prev_avoid = _prev_shock_avoid(y)

        # diag_pre_multiply(sigma, L) = diag(sigma) @ L
        scale_mat = sigma_logit_ab[:, None] * l_logit_ab  # (2, 2)
        logit_ab = mu_logit_ab[None, :] + z @ scale_mat  # (J, 2)
        a = jax.nn.sigmoid(logit_ab[:, 0])
        b = jax.nn.sigmoid(logit_ab[:, 1])

        p = (a[:, None] ** prev_shock) * (b[:, None] ** prev_avoid)  # (J, T)
        lp_lik = jnp.sum(tfd.Bernoulli(probs=p).log_prob(y))

        lp_prior = jnp.sum(tfd.Logistic(loc=0.0, scale=1.0).log_prob(mu_logit_ab))
        lp_prior = lp_prior + jnp.sum(tfd.Normal(loc=0.0, scale=1.0).log_prob(sigma_logit_ab))
        lp_prior = lp_prior + tfd.CholeskyLKJ(dimension=2, concentration=2.0).log_prob(l_logit_ab)
        lp_prior = lp_prior + jnp.sum(tfd.Normal(loc=0.0, scale=1.0).log_prob(z))

        return lp_lik + lp_prior
