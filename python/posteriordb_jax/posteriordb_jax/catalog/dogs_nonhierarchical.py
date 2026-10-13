"""dogs_nonhierarchical.stan (dog-specific a and b with correlated logits):

    parameters {
      vector[2] mu_logit_ab; vector<lower=0>[2] sigma_logit_ab;
      cholesky_factor_corr[2] L_logit_ab; matrix[J, 2] z;
    }
    transformed parameters {
      logit_ab = rep_vector(1, J) * mu_logit_ab' + z * diag_pre_multiply(sigma_logit_ab, L_logit_ab);
      a = inv_logit(logit_ab[:, 1]); b = inv_logit(logit_ab[:, 2]);
    }
    model {
      y[j, t] ~ bernoulli(a[j] ^ prev_shock[j, t] * b[j] ^ prev_avoid[j, t]);
      mu_logit_ab ~ logistic(0, 1);
      sigma_logit_ab ~ normal(0, 1);
      L_logit_ab ~ lkj_corr_cholesky(2);
      to_vector(z) ~ normal(0, 1);
    }

A plain Model: each dog has its own row of z. z is column-major in Stan's
free vector, hence order="F".
"""

import jax
import jax.numpy as jnp
from jax.scipy import stats

import modelgami as mg
import paragami as pg

from ._distributions import eval_lkj_corr_cholesky_logpdf
from ._dogs import eval_bernoulli_log_lik, count_previous

STAN_MODEL = "dogs_nonhierarchical"
POSTERIORS = ["dogs-dogs_nonhierarchical"]


def eval_log_prior(params, hyper):
    return (jnp.sum(stats.logistic.logpdf(params["mu_logit_ab"], loc=0.0, scale=1.0))
            + jnp.sum(stats.norm.logpdf(params["sigma_logit_ab"], loc=0.0, scale=1.0))
            + eval_lkj_corr_cholesky_logpdf(params["L_logit_ab"], eta=2.0)
            + jnp.sum(stats.norm.logpdf(params["z"], loc=0.0, scale=1.0)))


def eval_log_lik(data, params, hyper):
    scale = params["sigma_logit_ab"][:, None] * params["L_logit_ab"]   # diag_pre_multiply
    logit_ab = params["mu_logit_ab"][None, :] + params["z"] @ scale
    a, b = jax.nn.sigmoid(logit_ab[:, 0]), jax.nn.sigmoid(logit_ab[:, 1])
    shocks, avoids = count_previous(data["y"])
    p = a[:, None] ** shocks * b[:, None] ** avoids
    return jnp.sum(eval_bernoulli_log_lik(y=data["y"], p=p))


def build(raw):
    j = int(raw["n_dogs"])
    return mg.Model(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            mu_logit_ab=pg.NumericVectorPattern(2),
            sigma_logit_ab=pg.NumericVectorPattern(2, lb=0.0),
            L_logit_ab=pg.CholeskyCorrPattern(2),
            z=pg.NumericArrayPattern((j, 2), order="F"),
        ),
        prior=mg.DGP(eval_log_density=eval_log_prior),
        likelihood=mg.DGP(eval_log_density=eval_log_lik),
        data={"y": jnp.asarray(raw["y"], dtype=jnp.int32)},
    )
