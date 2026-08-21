"""JAX/TFP translation of posterior_database/models/stan/Rate_5_model.stan:

    parameters { real<lower=0, upper=1> theta; }
    model {
      theta ~ beta(1, 1);
      k1 ~ binomial(n1, theta);
      k2 ~ binomial(n2, theta);
    }
    generated quantities { ... }  // rng-only, not part of the log density
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class Rate5Model(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        return [ParameterSpec("theta", (), ConstraintType.LOWER_UPPER, bounds=(0.0, 1.0))]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        theta = params["theta"]
        lp = tfd.Beta(concentration1=1.0, concentration0=1.0).log_prob(theta)
        lp = lp + tfd.Binomial(total_count=float(data["n1"]), probs=theta).log_prob(float(data["k1"]))
        lp = lp + tfd.Binomial(total_count=float(data["n2"]), probs=theta).log_prob(float(data["k2"]))
        return lp
