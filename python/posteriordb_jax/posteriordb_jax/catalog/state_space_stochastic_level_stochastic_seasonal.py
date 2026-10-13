"""state_space_stochastic_level_stochastic_seasonal.stan (UK drivers series):

    parameters {
      vector<lower=mean(y) - 3 * sd(y), upper=mean(y) + 3 * sd(y)>[n] mu;
      vector[n] seasonal; real beta; real lambda; positive_ordered[3] sigma;
    }
    transformed parameters { yhat = mu + beta * x + lambda * w; }
    model {
      for (t in 12:n) seasonal[t] ~ normal(-sum(seasonal[t - 11:t - 1]), sigma[1]);
      for (t in 2:n) mu[t] ~ normal(mu[t - 1], sigma[2]);
      y ~ normal(yhat + seasonal, sigma[3]);
      sigma ~ student_t(4, 0, 1);
    }

mu's bounds depend only on the data, so they are computed in build(). The
seasonal and level terms are part of the prior. A plain Model.
"""

import jax.numpy as jnp
import numpy as np
from jax.scipy import stats

import modelgami as mg
import paragami as pg

STAN_MODEL = "state_space_stochastic_level_stochastic_seasonal"
POSTERIORS = ["uk_drivers-state_space_stochastic_level_stochastic_seasonal"]


def log_prior(params, hyper):
    seasonal, mu, sigma = params["seasonal"], params["mu"], params["sigma"]
    n = seasonal.shape[0]
    # sum of the 11 previous seasonal terms, for t = 12..n (1-based)
    previous_11 = jnp.stack([seasonal[11 - lag:n - lag] for lag in range(1, 12)]).sum(axis=0)
    return (jnp.sum(stats.norm.logpdf(seasonal[11:], loc=-previous_11, scale=sigma[0]))
            + jnp.sum(stats.norm.logpdf(mu[1:], loc=mu[:-1], scale=sigma[1]))
            + jnp.sum(stats.t.logpdf(sigma, df=4.0, loc=0.0, scale=1.0)))


def log_lik(params, data, hyper):
    yhat = params["mu"] + params["beta"] * data["x"] + params["lambda"] * data["w"]
    return jnp.sum(stats.norm.logpdf(data["y"], loc=yhat + params["seasonal"], scale=params["sigma"][2]))


def build(raw):
    y = np.asarray(raw["y"], dtype=float)
    n = int(raw["n"])
    center, spread = y.mean(), 3.0 * y.std(ddof=1)
    return mg.Model(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            mu=pg.NumericVectorPattern(n, lb=center - spread, ub=center + spread),
            seasonal=pg.NumericVectorPattern(n),
            beta=pg.NumericScalarPattern(),
            # `lambda` is a Python keyword, so this entry is passed as a dict.
            **{"lambda": pg.NumericScalarPattern()},
            sigma=pg.OrderedVectorPattern(3, positive=True),
        ),
        prior=mg.DGP(log_density=log_prior),
        likelihood=mg.DGP(log_density=log_lik),
        data={"y": raw["y"], "x": raw["x"], "w": raw["w"]},
    )
