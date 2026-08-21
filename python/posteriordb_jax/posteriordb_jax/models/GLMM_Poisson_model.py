"""JAX/TFP translation of posterior_database/models/stan/GLMM_Poisson_model.stan:

    parameters {
      real<lower=-20, upper=20> alpha; real<lower=-10, upper=10> beta1;
      real<lower=-10, upper=20> beta2; real<lower=-10, upper=10> beta3;
      vector[n] eps; real<lower=0, upper=5> sigma;
    }
    transformed parameters {
      log_lambda = alpha + beta1*year + beta2*year_squared + beta3*year_cubed + eps;
    }
    model {
      alpha ~ uniform(-20, 20); beta1 ~ uniform(-10, 10);
      beta2 ~ uniform(-10, 10); beta3 ~ uniform(-10, 10); sigma ~ uniform(0, 5);
      C ~ poisson_log(log_lambda);
      eps ~ normal(0, sigma);
    }

Every `uniform(a, b)` here IS a declared type-level bound too (`alpha`'s
prior interval `[-20,20]` matches its declared `<lower=-20,upper=20>`, same
for beta1/beta3/sigma) except `beta2`: declared `<lower=-10,upper=20>` but
given a *narrower* `uniform(-10, 10)` prior -- a real (if inconsequential
outside [-10,10], where its density would be `-inf`) mismatch in the
original Stan model between the type bound and the prior statement. Since
`uniform_lpdf`'s value is a nonzero constant (`-log(width)`) whenever
within its own bounds, all five are included explicitly for an exact
log-density match, using each prior's OWN stated interval (not the
possibly-wider type bound).
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class GLMMPoissonModel(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        n = int(data["n"])
        return [
            ParameterSpec("alpha", (), ConstraintType.LOWER_UPPER, bounds=(-20.0, 20.0)),
            ParameterSpec("beta1", (), ConstraintType.LOWER_UPPER, bounds=(-10.0, 10.0)),
            ParameterSpec("beta2", (), ConstraintType.LOWER_UPPER, bounds=(-10.0, 20.0)),
            ParameterSpec("beta3", (), ConstraintType.LOWER_UPPER, bounds=(-10.0, 10.0)),
            ParameterSpec("eps", (n,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("sigma", (), ConstraintType.LOWER_UPPER, bounds=(0.0, 5.0)),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        alpha = params["alpha"]
        beta1 = params["beta1"]
        beta2 = params["beta2"]
        beta3 = params["beta3"]
        eps = params["eps"]
        sigma = params["sigma"]

        year = data["year"]
        year_squared = year * year
        year_cubed = year * year * year
        log_lambda = alpha + beta1 * year + beta2 * year_squared + beta3 * year_cubed + eps

        c = jnp.asarray(data["C"], dtype=jnp.float64)

        lp = tfd.Uniform(low=-20.0, high=20.0).log_prob(alpha)
        lp = lp + tfd.Uniform(low=-10.0, high=10.0).log_prob(beta1)
        lp = lp + tfd.Uniform(low=-10.0, high=10.0).log_prob(beta2)
        lp = lp + tfd.Uniform(low=-10.0, high=10.0).log_prob(beta3)
        lp = lp + tfd.Uniform(low=0.0, high=5.0).log_prob(sigma)
        lp = lp + jnp.sum(tfd.Poisson(log_rate=log_lambda).log_prob(c))
        lp = lp + jnp.sum(tfd.Normal(loc=0.0, scale=sigma).log_prob(eps))
        return lp
