"""arK.stan (autoregressive model of order K):

    parameters { real alpha; array[K] real beta; real<lower=0> sigma; }
    model {
      alpha ~ normal(0, 10);
      beta ~ normal(0, 10);
      sigma ~ cauchy(0, 2.5);
      for (t in (K + 1):T)
        y[t] ~ normal(alpha + sum_k beta[k] * y[t - k], sigma);
    }

Not exchangeable: each observation depends on the K before it.
"""

import jax.numpy as jnp
from jax.scipy import stats

import modelgami as mg
import paragami as pg

STAN_MODEL = "arK"
POSTERIORS = ["arK-arK"]


def log_prior(params, hyper):
    return (stats.norm.logpdf(params["alpha"], loc=0.0, scale=10.0)
            + jnp.sum(stats.norm.logpdf(params["beta"], loc=0.0, scale=10.0))
            + stats.cauchy.logpdf(params["sigma"], loc=0.0, scale=2.5))


def log_lik(params, data, hyper):
    y, k = data["y"], params["beta"].shape[0]
    t = y.shape[0]
    # lags[:, j] = y[t - (j + 1)] for t = K..T-1 (0-based)
    lags = jnp.stack([y[k - j - 1:t - j - 1] for j in range(k)], axis=1)
    mean = params["alpha"] + lags @ params["beta"]
    return jnp.sum(stats.norm.logpdf(y[k:], loc=mean, scale=params["sigma"]))


def build(raw):
    k = int(raw["K"])
    return mg.Model(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            alpha=pg.NumericScalarPattern(),
            beta=pg.NumericVectorPattern(k),
            sigma=pg.NumericScalarPattern(lb=0.0),
        ),
        prior=mg.DGP(log_density=log_prior),
        likelihood=mg.DGP(log_density=log_lik),
        data={"y": raw["y"]},
    )
