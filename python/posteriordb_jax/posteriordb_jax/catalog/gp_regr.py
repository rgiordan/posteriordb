"""gp_regr.stan (Gaussian-process regression):

    parameters { real<lower=0> rho; real<lower=0> alpha; real<lower=0> sigma; }
    model {
      matrix[N, N] cov = gp_exp_quad_cov(x, alpha, rho) + diag_matrix(rep_vector(sigma, N));
      matrix[N, N] L_cov = cholesky_decompose(cov);
      rho ~ gamma(25, 4);
      alpha ~ normal(0, 2);
      sigma ~ normal(0, 1);
      y ~ multi_normal_cholesky(rep_vector(0, N), L_cov);
    }

Note that sigma (not sigma^2) is added to the diagonal, as in the Stan code.
Not exchangeable: the observations are jointly Gaussian.
"""

import jax.numpy as jnp
from jax.scipy import stats

import modelgami as mg
import paragami as pg

from ._gp import compute_exp_quad_cov

STAN_MODEL = "gp_regr"
POSTERIORS = ["gp_pois_regr-gp_regr"]


def eval_log_prior(params, hyper):
    return (stats.gamma.logpdf(params["rho"], a=25.0, scale=1.0 / 4.0)
            + stats.norm.logpdf(params["alpha"], loc=0.0, scale=2.0)
            + stats.norm.logpdf(params["sigma"], loc=0.0, scale=1.0))


def eval_log_lik(data, params, hyper):
    x, y = data["x"], data["y"]
    cov = compute_exp_quad_cov(x=x, alpha=params["alpha"], rho=params["rho"]) + params["sigma"] * jnp.eye(x.shape[0])
    return stats.multivariate_normal.logpdf(y, mean=jnp.zeros_like(y), cov=cov)


def build(raw):
    return mg.Model(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            rho=pg.NumericScalarPattern(lb=0.0),
            alpha=pg.NumericScalarPattern(lb=0.0),
            sigma=pg.NumericScalarPattern(lb=0.0),
        ),
        prior=mg.DGP(eval_log_density=eval_log_prior),
        likelihood=mg.DGP(eval_log_density=eval_log_lik),
        data={"x": raw["x"], "y": raw["y"]},
    )
