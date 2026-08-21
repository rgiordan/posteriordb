"""JAX/TFP translation of posterior_database/models/stan/surgical_model.stan:

    parameters { real mu; real<lower=0> sigmasq; array[N] real b; }
    transformed parameters { real<lower=0> sigma = sqrt(sigmasq); }
    model {
      mu ~ normal(0, 1000);
      sigmasq ~ inv_gamma(0.001, 0.001);
      b ~ normal(mu, sigma);
      r ~ binomial_logit(n, b);
    }

Stan's `inv_gamma(alpha, beta)` is shape/scale, matching `tfd.InverseGamma`'s
`concentration`/`scale` parameterization directly. (`generated quantities`'s
`pop_mean` is a derived RNG-free quantity that doesn't affect the log
density and is omitted.)
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class SurgicalModel(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        n = int(data["N"])
        return [
            ParameterSpec("mu", (), ConstraintType.UNCONSTRAINED),
            ParameterSpec("sigmasq", (), ConstraintType.LOWER, bounds=(0.0, None)),
            ParameterSpec("b", (n,), ConstraintType.UNCONSTRAINED),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        mu = params["mu"]
        sigmasq = params["sigmasq"]
        b = params["b"]

        sigma = jnp.sqrt(sigmasq)
        r = jnp.asarray(data["r"], dtype=jnp.float64)
        n = jnp.asarray(data["n"], dtype=jnp.float64)

        lp = tfd.Normal(loc=0.0, scale=1000.0).log_prob(mu)
        lp = lp + tfd.InverseGamma(concentration=0.001, scale=0.001).log_prob(sigmasq)
        lp = lp + jnp.sum(tfd.Normal(loc=mu, scale=sigma).log_prob(b))
        lp = lp + jnp.sum(tfd.Binomial(total_count=n, logits=b).log_prob(r))
        return lp
