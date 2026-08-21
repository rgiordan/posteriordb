"""JAX/TFP translation of posterior_database/models/stan/Rate_2_model.stan:

    parameters { real<lower=0, upper=1> theta1; real<lower=0, upper=1> theta2; }
    transformed parameters { real<lower=-1, upper=1> delta; delta = theta1 - theta2; }
    model {
      theta1 ~ beta(1, 1);
      theta2 ~ beta(1, 1);
      k1 ~ binomial(n1, theta1);
      k2 ~ binomial(n2, theta2);
    }

`delta` is a derived quantity with no prior of its own and isn't used
elsewhere in the model block, so it doesn't affect the log density and is
omitted here.
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class Rate2Model(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        return [
            ParameterSpec("theta1", (), ConstraintType.LOWER_UPPER, bounds=(0.0, 1.0)),
            ParameterSpec("theta2", (), ConstraintType.LOWER_UPPER, bounds=(0.0, 1.0)),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        theta1 = params["theta1"]
        theta2 = params["theta2"]
        beta_dist = tfd.Beta(concentration1=1.0, concentration0=1.0)
        lp = beta_dist.log_prob(theta1) + beta_dist.log_prob(theta2)
        lp = lp + tfd.Binomial(total_count=float(data["n1"]), probs=theta1).log_prob(float(data["k1"]))
        lp = lp + tfd.Binomial(total_count=float(data["n2"]), probs=theta2).log_prob(float(data["k2"]))
        return lp
