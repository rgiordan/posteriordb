"""JAX/TFP translation of posterior_database/models/stan/GLM_Poisson_model.stan:

    transformed data { year_squared = square(year); year_cubed = year_squared .* year; }
    parameters {
      real<lower=-20, upper=20> alpha;
      real<lower=-10, upper=10> beta1;
      real<lower=-10, upper=10> beta2;
      real<lower=-10, upper=10> beta3;
    }
    transformed parameters {
      vector[n] log_lambda = alpha + beta1*year + beta2*year_squared + beta3*year_cubed;
    }
    model {
      // Implicit uniform priors.
      C ~ poisson_log(log_lambda);
    }
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class GLMPoissonModel(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        return [
            ParameterSpec("alpha", (), ConstraintType.LOWER_UPPER, bounds=(-20.0, 20.0)),
            ParameterSpec("beta1", (), ConstraintType.LOWER_UPPER, bounds=(-10.0, 10.0)),
            ParameterSpec("beta2", (), ConstraintType.LOWER_UPPER, bounds=(-10.0, 10.0)),
            ParameterSpec("beta3", (), ConstraintType.LOWER_UPPER, bounds=(-10.0, 10.0)),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        alpha = params["alpha"]
        beta1 = params["beta1"]
        beta2 = params["beta2"]
        beta3 = params["beta3"]

        year = data["year"]
        year_squared = jnp.square(year)
        year_cubed = year_squared * year
        log_lambda = alpha + beta1 * year + beta2 * year_squared + beta3 * year_cubed

        c = jnp.asarray(data["C"], dtype=jnp.float64)
        return jnp.sum(tfd.Poisson(log_rate=log_lambda).log_prob(c))
