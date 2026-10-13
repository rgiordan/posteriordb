"""losscurve_sislob.stan (single-loss-ratio insurance loss development
curves: each cohort's cumulative losses follow its loss ratio times premium
times a growth curve in development time):

    parameters {
      real<lower=0> omega; real<lower=0> theta; vector<lower=0>[n_cohort] LR;
      real mu_LR; real<lower=0> sd_LR; real<lower=0> loss_sd;
    }
    transformed parameters {
      gf[i] = growthmodel_id == 1 ? 1 - exp(-(t_value[i] / theta)^omega)                  // Weibull
                                  : t_value[i]^omega / (t_value[i]^omega + theta^omega);  // log-logistic
      lm[i] = LR[cohort_id[i]] * premium[cohort_id[i]] * gf[t_idx[i]];
    }
    model {
      mu_LR ~ normal(0, 0.5); sd_LR ~ lognormal(0, 0.5); LR ~ lognormal(mu_LR, sd_LR);
      loss_sd ~ lognormal(0, 0.7); omega ~ lognormal(0, 0.5); theta ~ lognormal(0, 0.5);
      loss ~ normal(lm, (loss_sd * premium)[cohort_id]);
    }

Exchangeable: each observation carries its cohort and development-time
indices.
"""

import jax
import jax.numpy as jnp
from jax.scipy import stats

import modelgami as mg
import paragami as pg

from ._distributions import lognormal_logpdf

STAN_MODEL = "losscurve_sislob"
POSTERIORS = ["loss_curves-losscurve_sislob"]


def log_prior(params, hyper):
    return (stats.norm.logpdf(params["mu_LR"], loc=0.0, scale=0.5)
            + lognormal_logpdf(params["sd_LR"], mu=0.0, sigma=0.5)
            + jnp.sum(lognormal_logpdf(params["LR"], mu=params["mu_LR"], sigma=params["sd_LR"]))
            + lognormal_logpdf(params["loss_sd"], mu=0.0, sigma=0.7)
            + lognormal_logpdf(params["omega"], mu=0.0, sigma=0.5)
            + lognormal_logpdf(params["theta"], mu=0.0, sigma=0.5))


def growth_factor(*, t, omega, theta, weibull):
    t_omega = t**omega
    return jnp.where(weibull, 1.0 - jnp.exp(-((t / theta) ** omega)), t_omega / (t_omega + theta**omega))


def mean_and_sd(*, params, datum, hyper):
    cohort = datum["cohort_id"]
    premium = hyper["premium"][cohort]
    gf = growth_factor(t=hyper["t_value"][datum["t_idx"]], omega=params["omega"], theta=params["theta"],
                       weibull=hyper["growthmodel_id"] == 1)
    return params["LR"][cohort] * premium * gf, params["loss_sd"] * premium


def log_lik_one(params, datum, hyper):
    mean, sd = mean_and_sd(params=params, datum=datum, hyper=hyper)
    return stats.norm.logpdf(datum["loss"], loc=mean, scale=sd)


def sample_one(key, params, datum, hyper):
    mean, sd = mean_and_sd(params=params, datum=datum, hyper=hyper)
    return {"loss": mean + sd * jax.random.normal(key)}


def build(raw):
    return mg.ExchangeableModel(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            omega=pg.NumericScalarPattern(lb=0.0),
            theta=pg.NumericScalarPattern(lb=0.0),
            LR=pg.NumericVectorPattern(int(raw["n_cohort"]), lb=0.0),
            mu_LR=pg.NumericScalarPattern(),
            sd_LR=pg.NumericScalarPattern(lb=0.0),
            loss_sd=pg.NumericScalarPattern(lb=0.0),
        ),
        prior=mg.DGP(log_density=log_prior),
        likelihood=mg.ExchangeableDGP(log_density_one=log_lik_one, sample_one=sample_one),
        data={
            "loss": raw["loss"],
            "cohort_id": jnp.asarray(raw["cohort_id"], dtype=jnp.int32) - 1,
            "t_idx": jnp.asarray(raw["t_idx"], dtype=jnp.int32) - 1,
        },
        hyper={"t_value": raw["t_value"], "premium": raw["premium"], "growthmodel_id": int(raw["growthmodel_id"])},
    )
