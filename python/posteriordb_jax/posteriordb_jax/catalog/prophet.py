"""prophet.stan (Prophet's time-series model: a piecewise-linear trend with
changepoints, times additive and multiplicative regressors such as
seasonality):

    transformed data { A[i, s] = t[i] >= t_change[s]; }   // changepoint indicators
    parameters { real k; real m; vector[S] delta; real<lower=0> sigma_obs; vector[K] beta; }
    model {
      k ~ normal(0, 5); m ~ normal(0, 5); delta ~ double_exponential(0, tau);
      sigma_obs ~ normal(0, 0.5); beta ~ normal(0, sigmas);
      trend = (k + A * delta) .* t + (m + A * (-t_change .* delta));
      y ~ normal(trend .* (1 + X * (beta .* s_m)) + X * (beta .* s_a), sigma_obs);
    }

The data use the linear trend (trend_indicator = 0); the logistic trend is
not implemented. Exchangeable: each observation is its time, response and
row of regressors.
"""

import jax
import jax.numpy as jnp
from jax.scipy import stats

import modelgami as mg
import paragami as pg

STAN_MODEL = "prophet"
POSTERIORS = ["rstan_downloads-prophet"]


def log_prior(params, hyper):
    return (stats.norm.logpdf(params["k"], loc=0.0, scale=5.0)
            + stats.norm.logpdf(params["m"], loc=0.0, scale=5.0)
            + jnp.sum(stats.laplace.logpdf(params["delta"], loc=0.0, scale=hyper["tau"]))
            + stats.norm.logpdf(params["sigma_obs"], loc=0.0, scale=0.5)
            + jnp.sum(stats.norm.logpdf(params["beta"], loc=0.0, scale=hyper["sigmas"])))


def mean(*, params, datum, hyper):
    t, delta = datum["t"], params["delta"]
    after_change = t >= hyper["t_change"]   # this time's row of A
    trend = (params["k"] + after_change @ delta) * t + params["m"] - after_change @ (hyper["t_change"] * delta)
    x = datum["X"]
    return trend * (1.0 + x @ (params["beta"] * hyper["s_m"])) + x @ (params["beta"] * hyper["s_a"])


def log_lik_one(params, datum, hyper):
    return stats.norm.logpdf(datum["y"], loc=mean(params=params, datum=datum, hyper=hyper), scale=params["sigma_obs"])


def sample_one(key, params, datum, hyper):
    return {"y": mean(params=params, datum=datum, hyper=hyper) + params["sigma_obs"] * jax.random.normal(key)}


def build(raw):
    if int(raw["trend_indicator"]) != 0:
        raise NotImplementedError("only prophet's linear trend (trend_indicator = 0) is implemented")
    return mg.ExchangeableModel(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            k=pg.NumericScalarPattern(),
            m=pg.NumericScalarPattern(),
            delta=pg.NumericVectorPattern(int(raw["S"])),
            sigma_obs=pg.NumericScalarPattern(lb=0.0),
            beta=pg.NumericVectorPattern(int(raw["K"])),
        ),
        prior=mg.DGP(log_density=log_prior),
        likelihood=mg.ExchangeableDGP(log_density_one=log_lik_one, sample_one=sample_one),
        data={"t": raw["t"], "y": raw["y"], "X": raw["X"]},
        hyper={field: raw[field] for field in ("t_change", "tau", "sigmas", "s_a", "s_m")},
    )
