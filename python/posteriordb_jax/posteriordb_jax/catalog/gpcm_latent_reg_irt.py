"""gpcm_latent_reg_irt.stan (generalized partial credit model with a latent
regression of ability on person covariates):

    transformed data {
      m[i] = max response to item i; pos[i] = 1 + sum(m[1:i-1]);
      W_adj = centered and scaled W (obtain_adjustments);
    }
    parameters { vector<lower=0>[I] alpha; vector[sum(m) - 1] beta_free; vector[J] theta; vector[K] lambda_adj; }
    transformed parameters { vector[sum(m)] beta = append_row(beta_free, -sum(beta_free)); }
    model {
      alpha ~ lognormal(1, 1);
      beta ~ normal(0, 3);
      theta ~ normal(W_adj * lambda_adj, 1);
      lambda_adj ~ student_t(3, 0, 1);
      target += pcm(y[n], theta[jj[n]] * alpha[ii[n]], segment(beta, pos[ii[n]], m[ii[n]]));
    }

where pcm(y, eta, beta) = categorical_lpmf(y + 1 | softmax(cumulative_sum(append_row(0, eta - beta)))).
Items have different numbers of steps, so each response's step vector is
padded to the largest and masked. Not exchangeable in the modelgami sense:
each person and item has its own parameters.
"""

import jax.numpy as jnp
import numpy as np
from jax.scipy import stats

import modelgami as mg
import paragami as pg

from ._latent_reg_irt import (adjust_covariates, eval_categorical_steps_logpmf, eval_log_prior_alpha_lambda, sum_to_zero,
                              eval_theta_log_density)

STAN_MODEL = "gpcm_latent_reg_irt"
POSTERIORS = ["timssAusTwn_irt-gpcm_latent_reg_irt"]


def eval_log_prior(params, hyper):
    return (eval_log_prior_alpha_lambda(params)
            + jnp.sum(stats.norm.logpdf(sum_to_zero(params["beta_free"]), loc=0.0, scale=3.0)))


def eval_log_lik(data, params, hyper):
    ii = data["ii"]
    beta = sum_to_zero(params["beta_free"])
    eta = params["theta"][data["jj"]] * params["alpha"][ii]
    steps = eta[:, None] - beta[hyper["beta_index"][ii]]
    return (eval_theta_log_density(params=params, data=data)
            + jnp.sum(eval_categorical_steps_logpmf(y=data["y"], steps=steps, mask=hyper["step_mask"][ii])))


def build(raw):
    n_items, n_persons, n_covariates = int(raw["I"]), int(raw["J"]), int(raw["K"])
    ii = np.asarray(raw["ii"], dtype=int) - 1
    y = np.asarray(raw["y"], dtype=int)
    n_steps = np.zeros(n_items, dtype=int)
    np.maximum.at(n_steps, ii, y)
    first = np.concatenate([[0], np.cumsum(n_steps)[:-1]])
    step = np.arange(n_steps.max())
    step_mask = step[None, :] < n_steps[:, None]
    return mg.Model(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            alpha=pg.NumericVectorPattern(n_items, lb=0.0),
            beta_free=pg.NumericVectorPattern(int(n_steps.sum()) - 1),
            theta=pg.NumericVectorPattern(n_persons),
            lambda_adj=pg.NumericVectorPattern(n_covariates),
        ),
        prior=mg.DGP(eval_log_density=eval_log_prior),
        likelihood=mg.DGP(eval_log_density=eval_log_lik),
        data={
            "ii": jnp.asarray(ii, dtype=jnp.int32),
            "jj": jnp.asarray(raw["jj"], dtype=jnp.int32) - 1,
            "y": jnp.asarray(y, dtype=jnp.int32),
            "w_adj": adjust_covariates(raw["W"]),
        },
        hyper={
            "beta_index": jnp.asarray(np.where(step_mask, first[:, None] + step[None, :], 0), dtype=jnp.int32),
            "step_mask": jnp.asarray(step_mask),
        },
    )
