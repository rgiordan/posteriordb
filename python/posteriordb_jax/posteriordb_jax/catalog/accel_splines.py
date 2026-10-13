"""accel_splines.stan (brms distributional model of the mcycle data: the
mean and the log standard deviation are each a penalized spline):

    parameters {
      real Intercept; vector[Ks] bs; vector[knots_1] zs_1_1; real<lower=0> sds_1_1;
      real Intercept_sigma; vector[Ks_sigma] bs_sigma; vector[knots_sigma_1] zs_sigma_1_1; real<lower=0> sds_sigma_1_1;
    }
    model {
      mu = Intercept + Xs * bs + Zs_1_1 * (sds_1_1 * zs_1_1);
      sigma = exp(Intercept_sigma + Xs_sigma * bs_sigma + Zs_sigma_1_1 * (sds_sigma_1_1 * zs_sigma_1_1));
      Intercept ~ student_t(3, -13, 36);  zs_1_1 ~ normal(0, 1);  sds_1_1 ~ student_t(3, 0, 36) truncated at 0;
      Intercept_sigma ~ student_t(3, 0, 10);  zs_sigma_1_1 ~ normal(0, 1);  sds_sigma_1_1 ~ student_t(3, 0, 36) truncated at 0;
      Y ~ normal(mu, sigma);
    }

bs and bs_sigma have flat priors; the data have prior_only = 0. Exchangeable:
each observation is its response and its rows of the design matrices.
"""

import jax
import jax.numpy as jnp
from jax.scipy import stats

import modelgami as mg
import paragami as pg

STAN_MODEL = "accel_splines"
POSTERIORS = ["mcycle_splines-accel_splines"]


def eval_log_prior(params, hyper):
    # The half-Student-t priors on the sds add log(2): - student_t_lccdf(0 | 3, 0, 36).
    return (stats.t.logpdf(params["Intercept"], df=3.0, loc=-13.0, scale=36.0)
            + jnp.sum(stats.norm.logpdf(params["zs_1_1"], loc=0.0, scale=1.0))
            + stats.t.logpdf(params["sds_1_1"], df=3.0, loc=0.0, scale=36.0) + jnp.log(2.0)
            + stats.t.logpdf(params["Intercept_sigma"], df=3.0, loc=0.0, scale=10.0)
            + jnp.sum(stats.norm.logpdf(params["zs_sigma_1_1"], loc=0.0, scale=1.0))
            + stats.t.logpdf(params["sds_sigma_1_1"], df=3.0, loc=0.0, scale=36.0) + jnp.log(2.0))


def compute_mean_and_sd(*, params, datum):
    mu = (params["Intercept"] + datum["Xs"] @ params["bs"]
          + datum["Zs_1_1"] @ (params["sds_1_1"] * params["zs_1_1"]))
    log_sigma = (params["Intercept_sigma"] + datum["Xs_sigma"] @ params["bs_sigma"]
                 + datum["Zs_sigma_1_1"] @ (params["sds_sigma_1_1"] * params["zs_sigma_1_1"]))
    return mu, jnp.exp(log_sigma)


def eval_log_lik_one(datum, params, hyper):
    mu, sigma = compute_mean_and_sd(params=params, datum=datum)
    return stats.norm.logpdf(datum["Y"], loc=mu, scale=sigma)


def draw_one(key, datum, params, hyper):
    mu, sigma = compute_mean_and_sd(params=params, datum=datum)
    return {"Y": mu + sigma * jax.random.normal(key)}


def build(raw):
    return mg.ExchangeableModel(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            Intercept=pg.NumericScalarPattern(),
            bs=pg.NumericVectorPattern(int(raw["Ks"])),
            zs_1_1=pg.NumericVectorPattern(int(raw["knots_1"])),
            sds_1_1=pg.NumericScalarPattern(lb=0.0),
            Intercept_sigma=pg.NumericScalarPattern(),
            bs_sigma=pg.NumericVectorPattern(int(raw["Ks_sigma"])),
            zs_sigma_1_1=pg.NumericVectorPattern(int(raw["knots_sigma_1"])),
            sds_sigma_1_1=pg.NumericScalarPattern(lb=0.0),
        ),
        prior=mg.DGP(eval_log_density=eval_log_prior),
        likelihood=mg.ExchangeableDGP(eval_log_density_one=eval_log_lik_one, draw_one=draw_one),
        data={field: raw[field] for field in ("Y", "Xs", "Zs_1_1", "Xs_sigma", "Zs_sigma_1_1")},
    )
