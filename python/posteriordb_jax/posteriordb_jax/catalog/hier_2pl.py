"""hier_2pl.stan (hierarchical two-parameter logistic IRT with correlated item
parameters):

    parameters {
      vector[J] theta; vector[I] xi1; vector[I] xi2; vector[2] mu;
      vector<lower=0>[2] tau; cholesky_factor_corr[2] L_Omega;
    }
    transformed parameters { xi[i] = [xi1[i], xi2[i]]; alpha = exp(xi1); beta = xi2; }
    model {
      xi[i] ~ multi_normal_cholesky(mu, diag_pre_multiply(tau, L_Omega));
      theta ~ normal(0, 1);
      L_Omega ~ lkj_corr_cholesky(4);
      mu[1] ~ normal(0, 1); tau[1] ~ exponential(.1);
      mu[2] ~ normal(0, 5); tau[2] ~ exponential(.1);
      y ~ bernoulli_logit(alpha[ii] .* (theta[jj] - beta[ii]));
    }

A plain Model: each person has their own theta and each item its own xi.
"""

import jax
import jax.numpy as jnp
from jax.scipy import stats

import modelgami as mg
import paragami as pg

from ._distributions import lkj_corr_cholesky_logpdf

STAN_MODEL = "hier_2pl"
POSTERIORS = ["sat-hier_2pl"]


def log_prior(params, hyper):
    xi = jnp.stack([params["xi1"], params["xi2"]], axis=1)               # (I, 2)
    l_sigma = params["tau"][:, None] * params["L_Omega"]
    cov = l_sigma @ l_sigma.T
    mu, tau = params["mu"], params["tau"]
    return (jnp.sum(stats.multivariate_normal.logpdf(xi, mean=mu, cov=cov))
            + jnp.sum(stats.norm.logpdf(params["theta"], loc=0.0, scale=1.0))
            + lkj_corr_cholesky_logpdf(params["L_Omega"], eta=4.0)
            + stats.norm.logpdf(mu[0], loc=0.0, scale=1.0)
            + stats.expon.logpdf(tau[0], scale=10.0)
            + stats.norm.logpdf(mu[1], loc=0.0, scale=5.0)
            + stats.expon.logpdf(tau[1], scale=10.0))


def log_lik(params, data, hyper):
    item, person = data["ii"] - 1, data["jj"] - 1
    logit_p = jnp.exp(params["xi1"][item]) * (params["theta"][person] - params["xi2"][item])
    return jnp.sum(data["y"] * logit_p - jax.nn.softplus(logit_p))


def build(raw):
    i, j = int(raw["I"]), int(raw["J"])
    return mg.Model(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            theta=pg.NumericVectorPattern(j),
            xi1=pg.NumericVectorPattern(i),
            xi2=pg.NumericVectorPattern(i),
            mu=pg.NumericVectorPattern(2),
            tau=pg.NumericVectorPattern(2, lb=0.0),
            L_Omega=pg.CholeskyCorrPattern(2),
        ),
        prior=mg.DGP(log_density=log_prior),
        likelihood=mg.DGP(log_density=log_lik),
        data={key: jnp.asarray(raw[key], dtype=jnp.int32) for key in ("ii", "jj", "y")},
    )
