"""JAX/TFP translation of posterior_database/models/stan/dogs_log.stan:

    parameters { vector[2] beta; }
    transformed parameters {
      // n_avoid/n_shock: same cumulative-count-before-t recursion as `dogs`.
      p[j,t] = inv_logit(beta[1]*n_avoid[j,t] + beta[2]*n_shock[j,t]);
    }
    model {
      beta[1] ~ uniform(-100, 0);
      beta[2] ~ uniform(0, 100);
      y[i,j] ~ bernoulli(p[i,j]);
    }

Unlike `dogs`, `p` here is already a *probability* (via `inv_logit`), not a
logit, so the likelihood uses `tfd.Bernoulli(probs=p)`, not `logits=`.

IMPORTANT: `beta` is declared as a plain, fully unconstrained `vector[2]` --
`uniform(-100, 0)`/`uniform(0, 100)` are *prior sampling statements*, not
type-level bounds (unlike e.g. `real<lower=0,upper=1>`). Stan's sampler can
and does visit `beta` values outside those intervals; `uniform_lpdf` there
returns `-inf` (not a type error), which is exactly what happens for a
sizeable fraction of synthetic unconstrained draws. `tfd.Uniform.log_prob`
has the same "-inf outside [low, high]" behavior by construction, so this
matches Stan bit-for-bit as long as `beta` itself is declared UNCONSTRAINED
here too -- declaring it as a `LOWER_UPPER`-bounded `ParameterSpec` (an
earlier version of this file did, using a shared `[-100, 100]` bound) is
wrong on two counts: it applies a `Sigmoid` bijector Stan never applies
(so even the *constrained* value itself would disagree with Stan), and it
also can't express the two different intervals per component anyway.
"""

from __future__ import annotations

from typing import Dict, List

import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel
from .dogs import _n_avoid_shock

tfd = tfp.distributions


class DogsLog(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        return [ParameterSpec("beta", (2,), ConstraintType.UNCONSTRAINED)]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        beta = params["beta"]

        y = jnp.asarray(data["y"], dtype=jnp.float64)
        n_avoid, n_shock = _n_avoid_shock(y)
        p = jax.nn.sigmoid(beta[0] * n_avoid + beta[1] * n_shock)

        lp = tfd.Uniform(low=-100.0, high=0.0).log_prob(beta[0])
        lp = lp + tfd.Uniform(low=0.0, high=100.0).log_prob(beta[1])
        lp = lp + jnp.sum(tfd.Bernoulli(probs=p).log_prob(y))
        return lp
