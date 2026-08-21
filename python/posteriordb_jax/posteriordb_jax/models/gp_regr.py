"""JAX/TFP translation of posterior_database/models/stan/gp_regr.stan:

    parameters { real<lower=0> rho; real<lower=0> alpha; real<lower=0> sigma; }
    model {
      matrix[N, N] cov = gp_exp_quad_cov(x, alpha, rho) + diag_matrix(rep_vector(sigma, N));
      matrix[N, N] L_cov = cholesky_decompose(cov);
      rho ~ gamma(25, 4); alpha ~ normal(0, 2); sigma ~ normal(0, 1);
      y ~ multi_normal_cholesky(rep_vector(0, N), L_cov);
    }

Stan's `gp_exp_quad_cov(x, alpha, rho)[i,j] = alpha^2 * exp(-(x_i-x_j)^2 /
(2*rho^2))`, computed directly with `jnp.linalg.cholesky` (a fixed linear-
algebra step in the likelihood, not a `ParameterSpec` transform -- no
custom bijector needed here). Note the diagonal jitter is `sigma`, not
`sigma^2`, exactly as literally written in the Stan model.
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class GpRegr(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        return [
            ParameterSpec("rho", (), ConstraintType.LOWER, bounds=(0.0, None)),
            ParameterSpec("alpha", (), ConstraintType.LOWER, bounds=(0.0, None)),
            ParameterSpec("sigma", (), ConstraintType.LOWER, bounds=(0.0, None)),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        rho = params["rho"]
        alpha = params["alpha"]
        sigma = params["sigma"]

        x = data["x"]
        n = x.shape[0]
        dist2 = (x[:, None] - x[None, :]) ** 2
        cov = alpha**2 * jnp.exp(-0.5 * dist2 / rho**2) + sigma * jnp.eye(n, dtype=x.dtype)
        l_cov = jnp.linalg.cholesky(cov)

        lp = tfd.Gamma(concentration=25.0, rate=4.0).log_prob(rho)
        lp = lp + tfd.Normal(loc=0.0, scale=2.0).log_prob(alpha)
        lp = lp + tfd.Normal(loc=0.0, scale=1.0).log_prob(sigma)
        lp = lp + tfd.MultivariateNormalTriL(loc=jnp.zeros(n, dtype=x.dtype), scale_tril=l_cov).log_prob(
            data["y"]
        )
        return lp
