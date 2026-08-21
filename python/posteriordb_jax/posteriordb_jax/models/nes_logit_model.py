"""JAX/TFP translation of posterior_database/models/stan/nes_logit_model.stan:

    data {
      int<lower=0> N;
      vector[N] income;
      array[N] int<lower=0, upper=1> vote;
    }
    transformed data {
      matrix[N, 1] x = [income']';
    }
    parameters {
      real alpha;
      vector[1] beta;
    }
    model {
      vote ~ bernoulli_logit_glm(x, alpha, beta);
    }

`x` is just `income` reshaped to an (N, 1) matrix, so `x * beta == income *
beta[0]` -- the GLM likelihood is implemented directly with
`tfd.Bernoulli(logits=...)` rather than building the (unused) matrix.
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class NesLogitModel(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        return [
            ParameterSpec("alpha", (), ConstraintType.UNCONSTRAINED),
            ParameterSpec("beta", (1,), ConstraintType.UNCONSTRAINED),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        alpha = params["alpha"]
        beta = params["beta"]

        logits = alpha + beta[0] * data["income"]
        vote = jnp.asarray(data["vote"], dtype=jnp.float64)
        return jnp.sum(tfd.Bernoulli(logits=logits).log_prob(vote))
