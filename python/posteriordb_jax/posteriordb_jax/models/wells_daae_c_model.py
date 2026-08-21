"""JAX/TFP translation of posterior_database/models/stan/wells_daae_c_model.stan:

    transformed data {
      vector[N] c_dist100 = (dist - mean(dist)) / 100.0;
      vector[N] c_arsenic = arsenic - mean(arsenic);
      vector[N] da_inter = c_dist100 .* c_arsenic;
      vector[N] educ4 = educ / 4.0;
      matrix[N, 5] x = [c_dist100', c_arsenic', da_inter', assoc', educ4']';
    }
    parameters { real alpha; vector[5] beta; }
    model { switched ~ bernoulli_logit_glm(x, alpha, beta); }
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class WellsDaaeCModel(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        return [
            ParameterSpec("alpha", (), ConstraintType.UNCONSTRAINED),
            ParameterSpec("beta", (5,), ConstraintType.UNCONSTRAINED),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        alpha = params["alpha"]
        beta = params["beta"]
        dist = data["dist"]
        arsenic = data["arsenic"]
        c_dist100 = (dist - jnp.mean(dist)) / 100.0
        c_arsenic = arsenic - jnp.mean(arsenic)
        da_inter = c_dist100 * c_arsenic
        educ4 = data["educ"] / 4.0
        logits = (
            alpha
            + beta[0] * c_dist100
            + beta[1] * c_arsenic
            + beta[2] * da_inter
            + beta[3] * data["assoc"]
            + beta[4] * educ4
        )
        switched = jnp.asarray(data["switched"], dtype=jnp.float64)
        return jnp.sum(tfd.Bernoulli(logits=logits).log_prob(switched))
