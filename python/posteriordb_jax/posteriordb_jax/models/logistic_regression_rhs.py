"""JAX/TFP translation of posterior_database/models/stan/logistic_regression_rhs.stan
(the regularized horseshoe prior, Piironen & Vehtari 2017):

    parameters { real beta0; vector[d] z; real<lower=0> tau; vector<lower=0>[d] lambda; real<lower=0> caux; }
    transformed parameters {
      c = slab_scale * sqrt(caux);
      lambda_tilde = sqrt(c^2 * lambda^2 ./ (c^2 + tau^2 * lambda^2));
      beta = z .* lambda_tilde * tau;
    }
    model {
      z ~ std_normal();
      lambda ~ student_t(nu_local, 0, 1);
      tau ~ student_t(nu_global, 0, scale_global * 2);
      caux ~ inv_gamma(0.5 * slab_df, 0.5 * slab_df);
      beta0 ~ normal(0, scale_icept);
      y ~ bernoulli_logit_glm(x, beta0, beta);
    }

`nu_local`/`nu_global`/`scale_global`/`scale_icept`/`slab_scale`/`slab_df`
are fixed hyperparameters supplied as `data`, not tuned. None of the
half-t/`lambda`/`tau`/`caux` priors is explicitly truncated/renormalized in
Stan (despite `<lower=0>`), so each is a literal `student_t`/`inv_gamma`
density evaluated at the (already-constrained-positive) parameter value --
same convention as every other bounded-but-untruncated prior elsewhere in
this repo.
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions


class LogisticRegressionRhs(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        d = int(data["d"])
        return [
            ParameterSpec("beta0", (), ConstraintType.UNCONSTRAINED),
            ParameterSpec("z", (d,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("tau", (), ConstraintType.LOWER, bounds=(0.0, None)),
            ParameterSpec("lambda", (d,), ConstraintType.LOWER, bounds=(0.0, None)),
            ParameterSpec("caux", (), ConstraintType.LOWER, bounds=(0.0, None)),
        ]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        beta0 = params["beta0"]
        z = params["z"]
        tau = params["tau"]
        lam = params["lambda"]
        caux = params["caux"]

        slab_scale = data["slab_scale"]
        c = slab_scale * jnp.sqrt(caux)
        lambda_tilde = jnp.sqrt(c**2 * lam**2 / (c**2 + tau**2 * lam**2))
        beta = z * lambda_tilde * tau

        x = data["x"]
        logits = beta0 + x @ beta
        y = jnp.asarray(data["y"], dtype=jnp.float64)

        # TFP's StudentT infers its dtype from (df, loc, scale) at
        # construction time, *before* seeing the float64 sample passed to
        # .log_prob() -- bare Python floats/ints (as these data-supplied
        # hyperparameters are, straight out of JSON) get silently taken as
        # float32, which then raises a hard dtype-mismatch error against a
        # float64 sample (see diamonds.py's docstring for the same issue
        # with literal 3.0/8.0/10.0). Cast explicitly.
        f64 = lambda v: jnp.asarray(v, dtype=jnp.float64)  # noqa: E731
        nu_local = f64(data["nu_local"])
        nu_global = f64(data["nu_global"])
        scale_global = f64(data["scale_global"])
        scale_icept = f64(data["scale_icept"])
        slab_df = f64(data["slab_df"])

        lp = jnp.sum(tfd.Normal(loc=0.0, scale=1.0).log_prob(z))
        lp = lp + jnp.sum(tfd.StudentT(df=nu_local, loc=f64(0.0), scale=f64(1.0)).log_prob(lam))
        lp = lp + tfd.StudentT(df=nu_global, loc=f64(0.0), scale=scale_global * 2.0).log_prob(tau)
        lp = lp + tfd.InverseGamma(concentration=0.5 * slab_df, scale=0.5 * slab_df).log_prob(caux)
        lp = lp + tfd.Normal(loc=0.0, scale=scale_icept).log_prob(beta0)
        lp = lp + jnp.sum(tfd.Bernoulli(logits=logits).log_prob(y))
        return lp
