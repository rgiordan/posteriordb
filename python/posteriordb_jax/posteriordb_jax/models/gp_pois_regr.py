"""JAX/TFP translation of posterior_database/models/stan/gp_pois_regr.stan:

    parameters { real<lower=0> rho; real<lower=0> alpha; vector[N] f_tilde; }
    transformed parameters {
      matrix[N, N] cov = gp_exp_quad_cov(x, alpha, rho) + diag_matrix(rep_vector(1e-10, N));
      matrix[N, N] L_cov = cholesky_decompose(cov);
      f = L_cov * f_tilde;
    }
    model {
      rho ~ gamma(25, 4); alpha ~ normal(0, 2); f_tilde ~ normal(0, 1);
      k ~ poisson_log(f);
    }

Same GP kernel machinery as `gp_regr.py` (`gp_exp_quad_cov` + `jnp.linalg.cholesky`),
but here the covariance's jitter is a fixed `1e-10` (not a `sigma`
parameter), the Cholesky factor maps a whitened `f_tilde ~ normal(0,1)` to
the GP draw `f` (a `transformed parameters` reparameterization, not a
`multi_normal_cholesky` likelihood), and the observation likelihood is
`poisson_log(f)` (Poisson with log-rate `f`) rather than Gaussian.
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class GpPoisRegr(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        n = int(data["N"])
        return [
            ParameterSpec("rho", (), ConstraintType.LOWER, bounds=(0.0, None)),
            ParameterSpec("alpha", (), ConstraintType.LOWER, bounds=(0.0, None)),
            ParameterSpec("f_tilde", (n,), ConstraintType.UNCONSTRAINED),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        rho = params["rho"]
        alpha = params["alpha"]
        f_tilde = params["f_tilde"]

        x = jnp.asarray(data["x"], dtype=jnp.float64)
        k = jnp.asarray(data["k"], dtype=jnp.float64)
        n = x.shape[0]

        dist2 = (x[:, None] - x[None, :]) ** 2
        cov = alpha**2 * jnp.exp(-0.5 * dist2 / rho**2) + 1e-10 * jnp.eye(n, dtype=x.dtype)
        l_cov = jnp.linalg.cholesky(cov)
        f = l_cov @ f_tilde

        lp = tfd.Gamma(concentration=25.0, rate=4.0).log_prob(rho)
        lp = lp + tfd.Normal(loc=0.0, scale=2.0).log_prob(alpha)
        lp = lp + jnp.sum(tfd.Normal(loc=0.0, scale=1.0).log_prob(f_tilde))
        lp = lp + jnp.sum(tfd.Poisson(log_rate=f).log_prob(k))
        return lp
