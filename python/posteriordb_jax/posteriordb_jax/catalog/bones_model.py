"""bones_model.stan (graded-response model for skeletal maturity; gamma and
grade use -1 for missing):

    parameters { array[nChild] real theta; }
    model {
      theta ~ normal(0.0, 36);
      Q[i, j, k] = inv_logit(delta[j] * (theta[i] - gamma[j, k]));     // k < ncat[j]
      p[i, j, 1] = 1 - Q[i, j, 1];
      p[i, j, k] = Q[i, j, k - 1] - Q[i, j, k];                       // 1 < k < ncat[j]
      p[i, j, ncat[j]] = Q[i, j, ncat[j] - 1];
      if (grade[i, j] != -1) target += log(p[i, j, grade[i, j]]);
    }

Here Q is padded with Q[., ., 0] = 1 and Q[., ., k] = 0 for k >= ncat[j], so
every category is p[k] = Q[k - 1] - Q[k]. A plain Model: each child has its
own theta.
"""

import jax.numpy as jnp
from jax.scipy import stats
from jax.scipy.special import expit

import modelgami as mg
import paragami as pg

STAN_MODEL = "bones_model"
POSTERIORS = ["bones_data-bones_model"]


def log_prior(params, hyper):
    return jnp.sum(stats.norm.logpdf(params["theta"], loc=0.0, scale=36.0))


def log_lik(params, data, hyper):
    gamma, delta, ncat, grade = data["gamma"], data["delta"], data["ncat"], data["grade"]
    k = jnp.arange(1, gamma.shape[1] + 1)                          # thresholds 1..4
    has_threshold = k[None, :] < ncat[:, None]                     # (nInd, 4)
    q = expit(delta[None, :, None] * (params["theta"][:, None, None] - gamma[None, :, :]))
    q = jnp.where(has_threshold[None], q, 0.0)                     # (nChild, nInd, 4)
    ones = jnp.ones(q.shape[:2] + (1,))
    q_padded = jnp.concatenate([ones, q, jnp.zeros_like(ones)], axis=2)  # Q_0..Q_5
    p = q_padded[..., :-1] - q_padded[..., 1:]                     # categories 1..5
    observed = grade != -1
    g = jnp.where(observed, grade, 1) - 1
    log_p = jnp.log(jnp.take_along_axis(p, g[..., None], axis=2)[..., 0])
    return jnp.sum(jnp.where(observed, log_p, 0.0))


def build(raw):
    return mg.Model(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(theta=pg.NumericVectorPattern(int(raw["nChild"]))),
        prior=mg.DGP(log_density=log_prior),
        likelihood=mg.DGP(log_density=log_lik),
        data={
            "gamma": raw["gamma"],
            "delta": raw["delta"],
            "ncat": jnp.asarray(raw["ncat"], dtype=jnp.int32),
            "grade": jnp.asarray(raw["grade"], dtype=jnp.int32),
        },
    )
