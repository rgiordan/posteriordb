"""JAX/TFP translation of posterior_database/models/stan/GLM_Binomial_model.stan:

    transformed data { year_squared = year .* year; }
    parameters { real alpha; real beta1; real beta2; }
    transformed parameters { logit_p = alpha + beta1*year + beta2*year_squared; }
    model {
      alpha ~ normal(0, 100); beta1 ~ normal(0, 100); beta2 ~ normal(0, 100);
      C ~ binomial_logit(N, logit_p);
    }
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class GLMBinomialModel(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        return [
            ParameterSpec("alpha", (), ConstraintType.UNCONSTRAINED),
            ParameterSpec("beta1", (), ConstraintType.UNCONSTRAINED),
            ParameterSpec("beta2", (), ConstraintType.UNCONSTRAINED),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        alpha = params["alpha"]
        beta1 = params["beta1"]
        beta2 = params["beta2"]

        year = data["year"]
        year_squared = year * year
        logit_p = alpha + beta1 * year + beta2 * year_squared

        c = jnp.asarray(data["C"], dtype=jnp.float64)
        big_n = jnp.asarray(data["N"], dtype=jnp.float64)

        lp = tfd.Normal(loc=0.0, scale=100.0).log_prob(alpha)
        lp = lp + tfd.Normal(loc=0.0, scale=100.0).log_prob(beta1)
        lp = lp + tfd.Normal(loc=0.0, scale=100.0).log_prob(beta2)
        lp = lp + jnp.sum(tfd.Binomial(total_count=big_n, logits=logit_p).log_prob(c))
        return lp
