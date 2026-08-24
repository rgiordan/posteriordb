"""JAX/TFP translation of posterior_database/models/stan/election88_full.stan:

    parameters {
      vector[n_age] a; vector[n_edu] b; vector[n_age_edu] c; vector[n_state] d;
      vector[n_region_full] e; vector[5] beta;
      real<lower=0, upper=100> sigma_a; ... sigma_b; ... sigma_c; ... sigma_d; ... sigma_e;
    }
    transformed parameters {
      y_hat[i] = beta[1] + beta[2]*black[i] + beta[3]*female[i]
                 + beta[5]*female[i]*black[i] + beta[4]*v_prev_full[i]
                 + a[age[i]] + b[edu[i]] + c[age_edu[i]] + d[state[i]] + e[region_full[i]];
    }
    model {
      a ~ normal(0, sigma_a); b ~ normal(0, sigma_b); c ~ normal(0, sigma_c);
      d ~ normal(0, sigma_d); e ~ normal(0, sigma_e); beta ~ normal(0, 100);
      y ~ bernoulli_logit(y_hat);
    }

Mechanical hierarchical logistic regression: five group-level effects
(`a,b,c,d,e`), each with its own scale, summed into a linear predictor via
1-indexed (Stan) group indices (`age`, `edu`, `age_edu`, `state`,
`region_full`), same shape as the `radon_*`/`kidscore_*` families.
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class Election88Full(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        n_age = int(data["n_age"])
        n_edu = int(data["n_edu"])
        n_age_edu = int(data["n_age_edu"])
        n_state = int(data["n_state"])
        n_region_full = int(data["n_region_full"])
        return [
            ParameterSpec("a", (n_age,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("b", (n_edu,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("c", (n_age_edu,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("d", (n_state,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("e", (n_region_full,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("beta", (5,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("sigma_a", (), ConstraintType.LOWER_UPPER, bounds=(0.0, 100.0)),
            ParameterSpec("sigma_b", (), ConstraintType.LOWER_UPPER, bounds=(0.0, 100.0)),
            ParameterSpec("sigma_c", (), ConstraintType.LOWER_UPPER, bounds=(0.0, 100.0)),
            ParameterSpec("sigma_d", (), ConstraintType.LOWER_UPPER, bounds=(0.0, 100.0)),
            ParameterSpec("sigma_e", (), ConstraintType.LOWER_UPPER, bounds=(0.0, 100.0)),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        a = params["a"]
        b = params["b"]
        c = params["c"]
        d = params["d"]
        e = params["e"]
        beta = params["beta"]
        sigma_a = params["sigma_a"]
        sigma_b = params["sigma_b"]
        sigma_c = params["sigma_c"]
        sigma_d = params["sigma_d"]
        sigma_e = params["sigma_e"]

        age_idx = jnp.asarray(data["age"]) - 1
        edu_idx = jnp.asarray(data["edu"]) - 1
        age_edu_idx = jnp.asarray(data["age_edu"]) - 1
        state_idx = jnp.asarray(data["state"]) - 1
        region_full_idx = jnp.asarray(data["region_full"]) - 1
        black = jnp.asarray(data["black"], dtype=jnp.float64)
        female = jnp.asarray(data["female"], dtype=jnp.float64)
        v_prev_full = jnp.asarray(data["v_prev_full"], dtype=jnp.float64)
        y = jnp.asarray(data["y"], dtype=jnp.float64)

        y_hat = (
            beta[0]
            + beta[1] * black
            + beta[2] * female
            + beta[4] * female * black
            + beta[3] * v_prev_full
            + a[age_idx]
            + b[edu_idx]
            + c[age_edu_idx]
            + d[state_idx]
            + e[region_full_idx]
        )

        lp = jnp.sum(tfd.Normal(loc=0.0, scale=sigma_a).log_prob(a))
        lp = lp + jnp.sum(tfd.Normal(loc=0.0, scale=sigma_b).log_prob(b))
        lp = lp + jnp.sum(tfd.Normal(loc=0.0, scale=sigma_c).log_prob(c))
        lp = lp + jnp.sum(tfd.Normal(loc=0.0, scale=sigma_d).log_prob(d))
        lp = lp + jnp.sum(tfd.Normal(loc=0.0, scale=sigma_e).log_prob(e))
        lp = lp + jnp.sum(tfd.Normal(loc=0.0, scale=100.0).log_prob(beta))
        lp = lp + jnp.sum(tfd.Bernoulli(logits=y_hat).log_prob(y))
        return lp
