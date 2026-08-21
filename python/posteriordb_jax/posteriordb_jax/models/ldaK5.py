"""JAX/TFP translation of posterior_database/models/stan/ldaK5.stan:

    data { ... vector<lower=0>[5] alpha; vector<lower=0>[V] beta; }
    parameters {
      array[M] simplex[5] theta;
      array[5] simplex[V] phi;
    }
    model {
      theta[m] ~ dirichlet(alpha);
      phi[k] ~ dirichlet(beta);
      for (n in 1:N) {
        gamma[k] = log(theta[doc[n], k]) + log(phi[k, w[n]]);
        target += log_sum_exp(gamma);
      }
    }

Same structure as `ldaK2` (see that module for the array-of-simplex/gather
vectorization notes) with a fixed K=5 and `alpha`/`beta` supplied as data
rather than fixed to all-ones.
"""

from __future__ import annotations

from typing import Dict, List

import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class LdaK5(StatisticalModel):
    K = 5

    def param_specs(self, data) -> List[ParameterSpec]:
        v = int(data["V"])
        m = int(data["M"])
        k = self.K
        return [
            ParameterSpec("theta", (m, k), ConstraintType.SIMPLEX),
            ParameterSpec("phi", (k, v), ConstraintType.SIMPLEX),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        theta = params["theta"]  # (M, K)
        phi = params["phi"]  # (K, V)

        alpha = jnp.asarray(data["alpha"], dtype=jnp.float64)
        beta = jnp.asarray(data["beta"], dtype=jnp.float64)

        doc = jnp.asarray(data["doc"]) - 1  # (N,)
        w = jnp.asarray(data["w"]) - 1  # (N,)

        log_theta_n = jnp.log(theta)[doc]  # (N, K)
        log_phi_n = jnp.log(phi).T[w]  # (N, K)
        gamma = log_theta_n + log_phi_n  # (N, K)
        lp_lik = jnp.sum(jax.scipy.special.logsumexp(gamma, axis=-1))

        lp_prior = jnp.sum(tfd.Dirichlet(concentration=alpha).log_prob(theta))
        lp_prior = lp_prior + jnp.sum(tfd.Dirichlet(concentration=beta).log_prob(phi))
        return lp_lik + lp_prior
