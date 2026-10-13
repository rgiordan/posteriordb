"""grsm_latent_reg_irt.stan (generalized rating scale model with a latent
regression of ability on person covariates):

    transformed data { int m = max(y); W_adj = centered and scaled W (obtain_adjustments); }
    parameters {
      vector<lower=0>[I] alpha; vector[I - 1] beta_free; vector[m - 1] kappa_free;
      vector[J] theta; vector[K] lambda_adj;
    }
    transformed parameters {
      vector[I] beta = append_row(beta_free, -sum(beta_free));
      vector[m] kappa = append_row(kappa_free, -sum(kappa_free));
    }
    model {
      alpha ~ lognormal(1, 1);
      beta ~ normal(0, 3);
      kappa ~ normal(0, 3);
      theta ~ normal(W_adj * lambda_adj, 1);
      lambda_adj ~ student_t(3, 0, 1);
      target += rsm(y[n], theta[jj[n]] * alpha[ii[n]], beta[ii[n]], kappa);
    }

where rsm(y, eta, beta, kappa) = categorical_lpmf(y + 1 | softmax(cumulative_sum(append_row(0, eta - beta - kappa)))).
Not exchangeable in the modelgami sense: each person and item has its own
parameters.
"""

import jax.numpy as jnp
from jax.scipy import stats

import modelgami as mg
import paragami as pg

from ._latent_reg_irt import (adjusted_covariates, categorical_steps_logpmf, log_prior_alpha_lambda, sum_to_zero,
                              theta_log_density)

STAN_MODEL = "grsm_latent_reg_irt"
POSTERIORS = ["science_irt-grsm_latent_reg_irt"]


def log_prior(params, hyper):
    return (log_prior_alpha_lambda(params)
            + jnp.sum(stats.norm.logpdf(sum_to_zero(params["beta_free"]), loc=0.0, scale=3.0))
            + jnp.sum(stats.norm.logpdf(sum_to_zero(params["kappa_free"]), loc=0.0, scale=3.0)))


def log_lik(params, data, hyper):
    ii = data["ii"]
    eta = params["theta"][data["jj"]] * params["alpha"][ii] - sum_to_zero(params["beta_free"])[ii]
    steps = eta[:, None] - sum_to_zero(params["kappa_free"])[None, :]
    return (theta_log_density(params=params, data=data)
            + jnp.sum(categorical_steps_logpmf(y=data["y"], steps=steps)))


def build(raw):
    n_items, n_persons, n_covariates = int(raw["I"]), int(raw["J"]), int(raw["K"])
    y = jnp.asarray(raw["y"], dtype=jnp.int32)
    return mg.Model(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            alpha=pg.NumericVectorPattern(n_items, lb=0.0),
            beta_free=pg.NumericVectorPattern(n_items - 1),
            kappa_free=pg.NumericVectorPattern(int(y.max()) - 1),
            theta=pg.NumericVectorPattern(n_persons),
            lambda_adj=pg.NumericVectorPattern(n_covariates),
        ),
        prior=mg.DGP(log_density=log_prior),
        likelihood=mg.DGP(log_density=log_lik),
        data={
            "ii": jnp.asarray(raw["ii"], dtype=jnp.int32) - 1,
            "jj": jnp.asarray(raw["jj"], dtype=jnp.int32) - 1,
            "y": y,
            "w_adj": adjusted_covariates(raw["W"]),
        },
    )
