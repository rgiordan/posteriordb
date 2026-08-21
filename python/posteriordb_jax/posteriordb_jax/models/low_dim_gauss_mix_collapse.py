"""JAX/TFP translation of posterior_database/models/stan/low_dim_gauss_mix_collapse.stan:

Identical to `low_dim_gauss_mix` except `mu` is an unconstrained `vector[2]`
rather than an `ordered[2]` -- without the ordering constraint that resolves
the label-switching symmetry, hence "collapse" (both mixture-component
labelings are equally likely posterior modes).
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class LowDimGaussMixCollapse(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        return [
            ParameterSpec("mu", (2,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("sigma", (2,), ConstraintType.LOWER, bounds=(0.0, None)),
            ParameterSpec("theta", (), ConstraintType.LOWER_UPPER, bounds=(0.0, 1.0)),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        mu = params["mu"]
        sigma = params["sigma"]
        theta = params["theta"]

        y = data["y"]

        lp = jnp.sum(tfd.Normal(loc=0.0, scale=2.0).log_prob(sigma))
        lp = lp + jnp.sum(tfd.Normal(loc=0.0, scale=2.0).log_prob(mu))
        lp = lp + tfd.Beta(concentration1=5.0, concentration0=5.0).log_prob(theta)

        lp1 = tfd.Normal(loc=mu[0], scale=sigma[0]).log_prob(y)
        lp2 = tfd.Normal(loc=mu[1], scale=sigma[1]).log_prob(y)
        mix = jnp.logaddexp(jnp.log(theta) + lp1, jnp.log1p(-theta) + lp2)
        lp = lp + jnp.sum(mix)
        return lp
