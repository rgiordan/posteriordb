"""2pl_latent_reg_irt.stan (two-parameter logistic item response model with a
latent regression of ability on person covariates):

    transformed data { matrix[2, K] adj = obtain_adjustments(W); W_adj = (W - adj[1]) ./ adj[2]; }
    parameters { vector<lower=0>[I] alpha; vector[I - 1] beta_free; vector[J] theta; vector[K] lambda_adj; }
    transformed parameters { vector[I] beta = append_row(beta_free, -sum(beta_free)); }
    model {
      alpha ~ lognormal(1, 1);
      beta ~ normal(0, 3);
      lambda_adj ~ student_t(3, 0, 1);
      theta ~ normal(W_adj * lambda_adj, 1);
      y ~ bernoulli_logit(alpha[ii] .* theta[jj] - beta[ii]);
    }

Not exchangeable in the modelgami sense: each person and item has its own
parameters.
"""

import jax
import jax.numpy as jnp
from jax.scipy import stats

import modelgami as mg
import paragami as pg

from ._latent_reg_irt import adjusted_covariates, log_prior_alpha_lambda, sum_to_zero, theta_log_density

STAN_MODEL = "2pl_latent_reg_irt"
POSTERIORS = ["fims_Aus_Jpn_irt-2pl_latent_reg_irt"]


def log_prior(params, hyper):
    return (log_prior_alpha_lambda(params)
            + jnp.sum(stats.norm.logpdf(sum_to_zero(params["beta_free"]), loc=0.0, scale=3.0)))


def log_lik(params, data, hyper):
    ii = data["ii"]
    logit_p = params["alpha"][ii] * params["theta"][data["jj"]] - sum_to_zero(params["beta_free"])[ii]
    return (theta_log_density(params=params, data=data)
            + jnp.sum(data["y"] * logit_p - jax.nn.softplus(logit_p)))


def build(raw):
    n_items, n_persons, n_covariates = int(raw["I"]), int(raw["J"]), int(raw["K"])
    return mg.Model(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            alpha=pg.NumericVectorPattern(n_items, lb=0.0),
            beta_free=pg.NumericVectorPattern(n_items - 1),
            theta=pg.NumericVectorPattern(n_persons),
            lambda_adj=pg.NumericVectorPattern(n_covariates),
        ),
        prior=mg.DGP(log_density=log_prior),
        likelihood=mg.DGP(log_density=log_lik),
        data={
            "ii": jnp.asarray(raw["ii"], dtype=jnp.int32) - 1,
            "jj": jnp.asarray(raw["jj"], dtype=jnp.int32) - 1,
            "y": jnp.asarray(raw["y"], dtype=jnp.int32),
            "w_adj": adjusted_covariates(raw["W"]),
        },
    )
