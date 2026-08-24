"""JAX/TFP translation of posterior_database/models/stan/dogs_hierarchical.stan:

    data { array[n_dogs, n_trials] int<lower=0, upper=1> y; }
    transformed data {
      prev_shock[j,1] = 0; prev_avoid[j,1] = 0;
      prev_shock[j,t] = prev_shock[j,t-1] + y[j,t-1];
      prev_avoid[j,t] = prev_avoid[j,t-1] + 1 - y[j,t-1];
    }
    parameters { real<lower=0, upper=1> a; real<lower=0, upper=1> b; }
    model {
      p = a^prev_shock[j,t] * b^prev_avoid[j,t];
      y[j,t] ~ bernoulli(p);
    }

Same cumulative-count-before-t recursion as `dogs.py`'s `_n_avoid_shock`
(reused directly here under this model's own `prev_shock`/`prev_avoid`
names -- data-only, no `lax.scan` needed), but the parameters here are just
scalar `a, b in (0,1)` (no hierarchy, despite the model's name -- it's
"hierarchical" only in the sense that the per-trial detection probability
depends on the dog's own trial history, not that `a`/`b` themselves vary by
dog).
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


def _prev_shock_avoid(y: jnp.ndarray):
    """y: (n_dogs, n_trials) binary. Returns prev_shock, prev_avoid: (n_dogs,
    n_trials), each column t holding the cumulative count of shocks/avoids
    strictly before trial t (column 0 is all zeros)."""
    j = y.shape[0]
    zeros_col = jnp.zeros((j, 1), dtype=y.dtype)
    prev_shock = jnp.concatenate([zeros_col, jnp.cumsum(y[:, :-1], axis=1)], axis=1)
    prev_avoid = jnp.concatenate([zeros_col, jnp.cumsum(1 - y[:, :-1], axis=1)], axis=1)
    return prev_shock, prev_avoid


class DogsHierarchical(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        return [
            ParameterSpec("a", (), ConstraintType.LOWER_UPPER, bounds=(0.0, 1.0)),
            ParameterSpec("b", (), ConstraintType.LOWER_UPPER, bounds=(0.0, 1.0)),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        a = params["a"]
        b = params["b"]

        y = jnp.asarray(data["y"], dtype=jnp.float64)  # (n_dogs, n_trials)
        prev_shock, prev_avoid = _prev_shock_avoid(y)
        p = a**prev_shock * b**prev_avoid

        return jnp.sum(tfd.Bernoulli(probs=p).log_prob(y))
