"""JAX/TFP translation of posterior_database/models/stan/wells_dae_inter_model.stan:

    transformed data {
      vector[N] c_dist100 = (dist - mean(dist)) / 100.0;
      vector[N] c_arsenic = arsenic - mean(arsenic);
      vector[N] c_educ4 = (educ - mean(educ)) / 4.0;
      vector[N] da_inter = c_dist100 .* c_arsenic;
      vector[N] de_inter = c_dist100 .* c_educ4;
      vector[N] ae_inter = c_arsenic .* c_educ4;
      matrix[N, 6] x = [c_dist100', c_arsenic', c_educ4', da_inter', de_inter', ae_inter']';
    }
    parameters { real alpha; vector[6] beta; }
    model { switched ~ bernoulli_logit_glm(x, alpha, beta); }
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class WellsDaeInterModel(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        return [
            ParameterSpec("alpha", (), ConstraintType.UNCONSTRAINED),
            ParameterSpec("beta", (6,), ConstraintType.UNCONSTRAINED),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        alpha = params["alpha"]
        beta = params["beta"]
        dist = data["dist"]
        arsenic = data["arsenic"]
        educ = data["educ"]
        c_dist100 = (dist - jnp.mean(dist)) / 100.0
        c_arsenic = arsenic - jnp.mean(arsenic)
        c_educ4 = (educ - jnp.mean(educ)) / 4.0
        da_inter = c_dist100 * c_arsenic
        de_inter = c_dist100 * c_educ4
        ae_inter = c_arsenic * c_educ4
        logits = (
            alpha
            + beta[0] * c_dist100
            + beta[1] * c_arsenic
            + beta[2] * c_educ4
            + beta[3] * da_inter
            + beta[4] * de_inter
            + beta[5] * ae_inter
        )
        switched = jnp.asarray(data["switched"], dtype=jnp.float64)
        return jnp.sum(tfd.Bernoulli(logits=logits).log_prob(switched))
