"""JAX/TFP translation of posterior_database/models/stan/wells_dist.stan:

    parameters { vector[2] beta; }
    model { switched ~ bernoulli_logit(beta[1] + beta[2] * dist); }
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class WellsDist(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        return [ParameterSpec("beta", (2,), ConstraintType.UNCONSTRAINED)]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        beta = params["beta"]
        logits = beta[0] + beta[1] * data["dist"]
        switched = jnp.asarray(data["switched"], dtype=jnp.float64)
        return jnp.sum(tfd.Bernoulli(logits=logits).log_prob(switched))
