"""JAX/TFP translation of
posterior_database/models/stan/eight_schools_noncentered.stan:

    data {
      int<lower=0> J;
      array[J] real y;
      array[J] real<lower=0> sigma;
    }
    parameters {
      vector[J] theta_trans;
      real mu;
      real<lower=0> tau;
    }
    transformed parameters {
      vector[J] theta;
      theta = theta_trans * tau + mu;
    }
    model {
      theta_trans ~ normal(0, 1);
      y ~ normal(theta, sigma);
      mu ~ normal(0, 5);
      tau ~ cauchy(0, 5);
    }

NOTE: Stan's raw sampled parameter here is `theta_trans`, not `theta` --
posteriordb's reference draws for this posterior are keyed by the derived
`theta`, `mu`, `tau` (not `theta_trans`), so comparing against BridgeStan
needs an explicit `theta -> theta_trans` inverse (`stan_param_adapter` in
`testing.py`): `theta_trans = (theta - mu) / tau`.
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class EightSchoolsNoncentered(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        j = int(data["J"])
        return [
            ParameterSpec("theta_trans", (j,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("mu", (), ConstraintType.UNCONSTRAINED),
            ParameterSpec("tau", (), ConstraintType.LOWER, bounds=(0.0, None)),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        theta_trans = params["theta_trans"]
        mu = params["mu"]
        tau = params["tau"]
        theta = theta_trans * tau + mu

        y = data["y"]
        sigma = data["sigma"]

        lp = jnp.sum(tfd.Normal(loc=0.0, scale=1.0).log_prob(theta_trans))
        lp = lp + jnp.sum(tfd.Normal(loc=theta, scale=sigma).log_prob(y))
        lp = lp + tfd.Normal(loc=0.0, scale=5.0).log_prob(mu)
        lp = lp + tfd.Cauchy(loc=0.0, scale=5.0).log_prob(tau)
        return lp

    @staticmethod
    def stan_param_adapter(params: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        """Maps this model's constrained dict (keyed by `theta_trans`) to
        Stan's raw `parameters`-block constrained values (also keyed by
        `theta_trans`) -- identity here since our own param name already
        matches Stan's raw variable name; `theta` (the transformed
        parameter posteriordb's draws are keyed by) is handled separately
        in the testing harness via `theta_to_theta_trans`."""
        return params

    @staticmethod
    def theta_trans_from_theta(theta, mu, tau):
        return (theta - mu) / tau
