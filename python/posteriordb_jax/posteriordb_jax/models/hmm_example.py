"""JAX/TFP translation of posterior_database/models/stan/hmm_example.stan:
a 2-state Gaussian-emission HMM with a forward-algorithm likelihood.

    parameters {
      simplex[K] theta1;   // transitions out of state 1
      simplex[K] theta2;   // transitions out of state 2
      positive_ordered[K] mu;
    }
    model {
      target += normal_lpdf(mu[1] | 3, 1);
      target += normal_lpdf(mu[2] | 10, 1);
      // forward algorithm over y, using theta = [theta1; theta2] as the
      // K x K transition matrix and Normal(mu[k], 1) emissions
    }

NOTE: the model itself only ever references `mu[1]`/`mu[2]` and exactly two
transition-row parameters (`theta1`, `theta2`) -- it is hardcoded to K=2
despite the generic-looking `for k in 1:K` loops (posteriordb's own
`hmm_example` data has K=2). This translation makes the same K=2 assumption
rather than generalizing beyond what the Stan source itself supports.
"""

from __future__ import annotations

from typing import Dict, List

import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class HmmExample(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        k = int(data["K"])
        assert k == 2, "hmm_example's Stan source hardcodes K=2 (mu[1]/mu[2] priors, two theta rows)"
        return [
            ParameterSpec("theta1", (k,), ConstraintType.SIMPLEX),
            ParameterSpec("theta2", (k,), ConstraintType.SIMPLEX),
            ParameterSpec("mu", (k,), ConstraintType.POSITIVE_ORDERED),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        theta1 = params["theta1"]
        theta2 = params["theta2"]
        mu = params["mu"]
        y = data["y"]

        lp_prior = tfd.Normal(loc=3.0, scale=1.0).log_prob(mu[0]) + tfd.Normal(
            loc=10.0, scale=1.0
        ).log_prob(mu[1])

        log_theta = jnp.log(jnp.stack([theta1, theta2], axis=0))  # log_theta[j, k]
        emission = tfd.Normal(loc=mu[None, :], scale=1.0).log_prob(y[:, None])  # (N, K)

        gamma1 = emission[0]

        def step(gamma_prev, emission_t):
            acc = gamma_prev[:, None] + log_theta  # acc[j, k]
            gamma_t = jax.scipy.special.logsumexp(acc, axis=0) + emission_t
            return gamma_t, None

        gamma_final, _ = jax.lax.scan(step, gamma1, emission[1:])
        lp_forward = jax.scipy.special.logsumexp(gamma_final)

        return lp_prior + lp_forward
