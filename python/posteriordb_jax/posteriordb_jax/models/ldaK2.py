"""JAX/TFP translation of posterior_database/models/stan/ldaK2.stan:

    transformed data { K=2; alpha = ones(K); beta = ones(V); }
    parameters {
      array[M] simplex[K] theta; // topic dist for doc m
      array[K] simplex[V] phi;   // word dist for topic k
    }
    model {
      theta[m] ~ dirichlet(alpha);
      phi[k] ~ dirichlet(beta);
      for (n in 1:N) {
        gamma[k] = log(theta[doc[n], k]) + log(phi[k, w[n]]);
        target += log_sum_exp(gamma);
      }
    }

First model exercising `ParameterSpec`'s "array of simplex" support
(`theta`: `array[M] simplex[K]`, `phi`: `array[K] simplex[V]`) -- see
`constraints.py`'s `ParameterSpec.free_shape`/`.free_flatten_order`.

`w`/`doc` are Stan's 1-indexed word/doc-id arrays; converted to 0-indexed
gathers. The per-word mixture log-density is fully vectorized: `theta[doc-1]`
gathers each word's document-topic row (N,K), `phi.T[w-1]` gathers each
word's topic-word column (N,K), and `log_sum_exp` reduces over K per word.
"""

from __future__ import annotations

from typing import Dict, List

import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class LdaK2(StatisticalModel):
    K = 2

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

        v = int(data["V"])
        k = self.K
        alpha = jnp.ones((k,), dtype=jnp.float64)
        beta = jnp.ones((v,), dtype=jnp.float64)

        doc = jnp.asarray(data["doc"]) - 1  # (N,)
        w = jnp.asarray(data["w"]) - 1  # (N,)

        log_theta_n = jnp.log(theta)[doc]  # (N, K)
        log_phi_n = jnp.log(phi).T[w]  # (N, K)
        gamma = log_theta_n + log_phi_n  # (N, K)
        lp_lik = jnp.sum(jax.scipy.special.logsumexp(gamma, axis=-1))

        lp_prior = jnp.sum(tfd.Dirichlet(concentration=alpha).log_prob(theta))
        lp_prior = lp_prior + jnp.sum(tfd.Dirichlet(concentration=beta).log_prob(phi))
        return lp_lik + lp_prior
