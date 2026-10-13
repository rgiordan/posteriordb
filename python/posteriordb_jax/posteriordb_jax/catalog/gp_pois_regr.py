"""gp_pois_regr.stan (latent Gaussian-process Poisson regression, whitened):

    parameters { real<lower=0> rho; real<lower=0> alpha; vector[N] f_tilde; }
    transformed parameters {
      cov = gp_exp_quad_cov(x, alpha, rho) + diag_matrix(rep_vector(1e-10, N));
      f = cholesky_decompose(cov) * f_tilde;
    }
    model {
      rho ~ gamma(25, 4);
      alpha ~ normal(0, 2);
      f_tilde ~ normal(0, 1);
      k ~ poisson_log(f);
    }

Not exchangeable: each observation has its own latent f_tilde.
"""

import jax.numpy as jnp
from jax.scipy import stats

import modelgami as mg
import paragami as pg

from ._gp import compute_exp_quad_cov

STAN_MODEL = "gp_pois_regr"
POSTERIORS = ["gp_pois_regr-gp_pois_regr"]


def eval_log_prior(params, hyper):
    return (stats.gamma.logpdf(params["rho"], a=25.0, scale=1.0 / 4.0)
            + stats.norm.logpdf(params["alpha"], loc=0.0, scale=2.0)
            + jnp.sum(stats.norm.logpdf(params["f_tilde"], loc=0.0, scale=1.0)))


def compute_f(params, x):
    """The transformed parameter f = cholesky(cov) @ f_tilde."""
    cov = compute_exp_quad_cov(x=x, alpha=params["alpha"], rho=params["rho"]) + 1e-10 * jnp.eye(x.shape[0])
    return jnp.linalg.cholesky(cov) @ params["f_tilde"]


def eval_log_lik(data, params, hyper):
    log_rate = compute_f(params=params, x=data["x"])
    return jnp.sum(stats.poisson.logpmf(data["k"], mu=jnp.exp(log_rate)))


def build(raw):
    n = int(raw["N"])
    return mg.Model(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            rho=pg.NumericScalarPattern(lb=0.0),
            alpha=pg.NumericScalarPattern(lb=0.0),
            f_tilde=pg.NumericVectorPattern(n),
        ),
        prior=mg.DGP(eval_log_density=eval_log_prior),
        likelihood=mg.DGP(eval_log_density=eval_log_lik),
        data={"x": raw["x"], "k": jnp.asarray(raw["k"], dtype=jnp.int32)},
    )
