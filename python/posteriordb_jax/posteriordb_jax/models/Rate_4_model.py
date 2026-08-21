"""JAX/TFP translation of posterior_database/models/stan/Rate_4_model.stan:

    parameters { real<lower=0, upper=1> theta; real<lower=0, upper=1> thetaprior; }
    model {
      theta ~ beta(1, 1);
      thetaprior ~ beta(1, 1);
      k ~ binomial(n, theta);
    }
    generated quantities { ... }  // rng-only, not part of the log density

`thetaprior` has a prior but never enters the likelihood -- it's Stan's
prior-predictive scaffold (see `generated quantities`), included here only
for its (data-independent) prior term since it's a free parameter with its
own Jacobian contribution.
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class Rate4Model(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        return [
            ParameterSpec("theta", (), ConstraintType.LOWER_UPPER, bounds=(0.0, 1.0)),
            ParameterSpec("thetaprior", (), ConstraintType.LOWER_UPPER, bounds=(0.0, 1.0)),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        theta = params["theta"]
        thetaprior = params["thetaprior"]
        beta_dist = tfd.Beta(concentration1=1.0, concentration0=1.0)
        lp = beta_dist.log_prob(theta) + beta_dist.log_prob(thetaprior)
        lp = lp + tfd.Binomial(total_count=float(data["n"]), probs=theta).log_prob(float(data["k"]))
        return lp
