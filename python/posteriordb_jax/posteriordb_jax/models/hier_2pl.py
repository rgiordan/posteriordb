"""JAX/TFP translation of posterior_database/models/stan/hier_2pl.stan:

    parameters {
      vector[J] theta; vector[I] xi1; vector[I] xi2;
      vector[2] mu; vector<lower=0>[2] tau; cholesky_factor_corr[2] L_Omega;
    }
    transformed parameters {
      xi[i,1] = xi1[i]; xi[i,2] = xi2[i]; alpha[i] = exp(xi[i,1]); beta[i] = xi[i,2];
    }
    model {
      L_Sigma = diag_pre_multiply(tau, L_Omega);
      xi[i] ~ multi_normal_cholesky(mu, L_Sigma);
      theta ~ normal(0, 1);
      L_Omega ~ lkj_corr_cholesky(4);
      mu[1] ~ normal(0, 1); tau[1] ~ exponential(.1);
      mu[2] ~ normal(0, 5); tau[2] ~ exponential(.1);
      y ~ bernoulli_logit(alpha[ii] .* (theta[jj] - beta[ii]));
    }

`xi[i] = (xi1[i], xi2[i])` is stacked into an `(I, 2)` array and its
`multi_normal_cholesky` prior evaluated in one batched
`tfd.MultivariateNormalTriL` call (broadcasting `mu`/`L_Sigma` over the `I`
items) rather than a per-item Python loop. (`Omega` in `generated
quantities` is a derived, RNG-free quantity and is omitted.)
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class Hier2pl(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        i = int(data["I"])
        j = int(data["J"])
        return [
            ParameterSpec("theta", (j,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("xi1", (i,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("xi2", (i,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("mu", (2,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("tau", (2,), ConstraintType.LOWER, bounds=(0.0, None)),
            ParameterSpec("L_Omega", (2, 2), ConstraintType.CHOLESKY_FACTOR_CORR),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        theta = params["theta"]
        xi1 = params["xi1"]
        xi2 = params["xi2"]
        mu = params["mu"]
        tau = params["tau"]
        l_omega = params["L_Omega"]

        xi = jnp.stack([xi1, xi2], axis=-1)  # (I, 2)
        alpha = jnp.exp(xi1)
        beta = xi2

        l_sigma = tau[:, None] * l_omega  # diag_pre_multiply(tau, L_Omega)

        ii = jnp.asarray(data["ii"]) - 1
        jj = jnp.asarray(data["jj"]) - 1
        logits = alpha[ii] * (theta[jj] - beta[ii])
        y = jnp.asarray(data["y"], dtype=jnp.float64)

        lp = jnp.sum(
            tfd.MultivariateNormalTriL(loc=mu, scale_tril=l_sigma).log_prob(xi)
        )
        lp = lp + jnp.sum(tfd.Normal(loc=0.0, scale=1.0).log_prob(theta))
        lp = lp + tfd.CholeskyLKJ(dimension=2, concentration=4.0).log_prob(l_omega)
        lp = lp + tfd.Normal(loc=0.0, scale=1.0).log_prob(mu[0])
        lp = lp + tfd.Exponential(rate=0.1).log_prob(tau[0])
        lp = lp + tfd.Normal(loc=0.0, scale=5.0).log_prob(mu[1])
        lp = lp + tfd.Exponential(rate=0.1).log_prob(tau[1])
        lp = lp + jnp.sum(tfd.Bernoulli(logits=logits).log_prob(y))
        return lp
