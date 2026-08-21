"""JAX/TFP translation of posterior_database/models/stan/wells_interaction_model.stan:

    transformed data {
      vector[N] dist100 = dist / 100.0;
      vector[N] inter = dist100 .* arsenic;
      matrix[N, 3] x = [dist100', arsenic', inter']';
    }
    parameters { real alpha; vector[3] beta; }
    model { switched ~ bernoulli_logit_glm(x, alpha, beta); }
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class WellsInteractionModel(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        return [
            ParameterSpec("alpha", (), ConstraintType.UNCONSTRAINED),
            ParameterSpec("beta", (3,), ConstraintType.UNCONSTRAINED),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        alpha = params["alpha"]
        beta = params["beta"]
        dist100 = data["dist"] / 100.0
        arsenic = data["arsenic"]
        inter = dist100 * arsenic
        logits = alpha + beta[0] * dist100 + beta[1] * arsenic + beta[2] * inter
        switched = jnp.asarray(data["switched"], dtype=jnp.float64)
        return jnp.sum(tfd.Bernoulli(logits=logits).log_prob(switched))
