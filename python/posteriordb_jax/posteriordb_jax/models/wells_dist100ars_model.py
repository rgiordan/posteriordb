"""JAX/TFP translation of posterior_database/models/stan/wells_dist100ars_model.stan:

    transformed data {
      vector[N] dist100 = dist / 100.0;
      matrix[N, 2] x = [dist100', arsenic']';
    }
    parameters { real alpha; vector[2] beta; }
    model { switched ~ bernoulli_logit_glm(x, alpha, beta); }
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class WellsDist100arsModel(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        return [
            ParameterSpec("alpha", (), ConstraintType.UNCONSTRAINED),
            ParameterSpec("beta", (2,), ConstraintType.UNCONSTRAINED),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        alpha = params["alpha"]
        beta = params["beta"]
        dist100 = data["dist"] / 100.0
        logits = alpha + beta[0] * dist100 + beta[1] * data["arsenic"]
        switched = jnp.asarray(data["switched"], dtype=jnp.float64)
        return jnp.sum(tfd.Bernoulli(logits=logits).log_prob(switched))
