"""JAX/TFP translation of posterior_database/models/stan/bym2_offset_only.stan:

    parameters {
      real beta0; real<lower=0> sigma; real<lower=0, upper=1> rho;
      vector[N] theta; vector[N] phi;
    }
    transformed parameters {
      convolved_re = sqrt(1 - rho) * theta + sqrt(rho / scaling_factor) * phi;
    }
    model {
      y ~ poisson_log(log_E + beta0 + convolved_re * sigma);
      target += -0.5 * dot_self(phi[node1] - phi[node2]);
      beta0 ~ normal(0, 1); theta ~ normal(0, 1); sigma ~ normal(0, 1); rho ~ beta(0.5, 0.5);
      sum(phi) ~ normal(0, 0.001 * N);
    }

The pairwise ICAR prior term (`-0.5 * dot_self(phi[node1] - phi[node2])`,
an *improper* prior over the edge graph, not a proper distribution with its
own `ParameterSpec`) and the soft sum-to-zero constraint on `phi` are both
literal `target +=` statements, translated directly. `node1`/`node2` are
Stan's 1-indexed edge endpoints.
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class Bym2OffsetOnly(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        n = int(data["N"])
        return [
            ParameterSpec("beta0", (), ConstraintType.UNCONSTRAINED),
            ParameterSpec("sigma", (), ConstraintType.LOWER, bounds=(0.0, None)),
            ParameterSpec("rho", (), ConstraintType.LOWER_UPPER, bounds=(0.0, 1.0)),
            ParameterSpec("theta", (n,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("phi", (n,), ConstraintType.UNCONSTRAINED),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        beta0 = params["beta0"]
        sigma = params["sigma"]
        rho = params["rho"]
        theta = params["theta"]
        phi = params["phi"]

        n = int(data["N"])
        scaling_factor = data["scaling_factor"]
        convolved_re = jnp.sqrt(1.0 - rho) * theta + jnp.sqrt(rho / scaling_factor) * phi

        log_e = jnp.log(data["E"])
        eta = log_e + beta0 + convolved_re * sigma
        y = jnp.asarray(data["y"], dtype=jnp.float64)

        node1 = jnp.asarray(data["node1"]) - 1
        node2 = jnp.asarray(data["node2"]) - 1
        icar_term = -0.5 * jnp.sum((phi[node1] - phi[node2]) ** 2)

        lp = jnp.sum(tfd.Poisson(log_rate=eta).log_prob(y))
        lp = lp + icar_term
        lp = lp + tfd.Normal(loc=0.0, scale=1.0).log_prob(beta0)
        lp = lp + jnp.sum(tfd.Normal(loc=0.0, scale=1.0).log_prob(theta))
        lp = lp + tfd.Normal(loc=0.0, scale=1.0).log_prob(sigma)
        lp = lp + tfd.Beta(concentration1=0.5, concentration0=0.5).log_prob(rho)
        lp = lp + tfd.Normal(loc=0.0, scale=0.001 * n).log_prob(jnp.sum(phi))
        return lp
