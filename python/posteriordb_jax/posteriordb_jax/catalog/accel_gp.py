"""accel_gp.stan (brms distributional model of the mcycle data: the mean and
the log standard deviation are each an intercept plus an approximate
(Hilbert-space basis) Gaussian process):

    parameters {
      real Intercept; real<lower=0> sdgp_1; real<lower=0> lscale_1; vector[NBgp_1] zgp_1;
      real Intercept_sigma; real<lower=0> sdgp_sigma_1; real<lower=0> lscale_sigma_1; vector[NBgp_sigma_1] zgp_sigma_1;
    }
    model {
      mu = Intercept + gpa(Xgp_1, sdgp_1, lscale_1, zgp_1, slambda_1);
      sigma = exp(Intercept_sigma + gpa(Xgp_sigma_1, sdgp_sigma_1, lscale_sigma_1, zgp_sigma_1, slambda_sigma_1));
      Intercept ~ student_t(3, -13, 36);  sdgp_1 ~ student_t(3, 0, 36) truncated at 0;
      zgp_1 ~ normal(0, 1);  lscale_1 ~ inv_gamma(1.124909, 0.0177);
      Intercept_sigma ~ student_t(3, 0, 10);  sdgp_sigma_1 ~ student_t(3, 0, 36) truncated at 0;
      zgp_sigma_1 ~ normal(0, 1);  lscale_sigma_1 ~ inv_gamma(1.124909, 0.0177);
      Y ~ normal(mu, sigma);
    }

where gpa(X, sdgp, lscale, z, slambda) = X * (sqrt(spd(slambda)) .* z) and,
for one input dimension, spd(s) = sdgp^2 * sqrt(2 pi) * lscale * exp(-lscale^2 s^2 / 2).
The data have Dgp = 1 and prior_only = 0. Exchangeable: each observation is
its response and its rows of the two basis matrices.
"""

import jax
import jax.numpy as jnp
from jax.scipy import stats

import modelgami as mg
import paragami as pg

from ._distributions import inv_gamma_logpdf

STAN_MODEL = "accel_gp"
POSTERIORS = ["mcycle_gp-accel_gp"]


def gp_weights(*, sdgp, lscale, z, slambda):
    """sqrt(spd(slambda)) .* z, the basis coefficients of the approximate GP.
    The square root is taken analytically so its gradient stays finite where
    spd underflows to zero."""
    sqrt_spd = sdgp * (2.0 * jnp.pi) ** 0.25 * jnp.sqrt(lscale) * jnp.exp(-0.25 * lscale**2 * slambda**2)
    return sqrt_spd * z


def log_prior(params, hyper):
    # The half-Student-t priors on the sds add log(2): - student_t_lccdf(0 | 3, 0, 36).
    return (stats.t.logpdf(params["Intercept"], df=3.0, loc=-13.0, scale=36.0)
            + stats.t.logpdf(params["sdgp_1"], df=3.0, loc=0.0, scale=36.0) + jnp.log(2.0)
            + jnp.sum(stats.norm.logpdf(params["zgp_1"], loc=0.0, scale=1.0))
            + inv_gamma_logpdf(params["lscale_1"], a=1.124909, b=0.0177)
            + stats.t.logpdf(params["Intercept_sigma"], df=3.0, loc=0.0, scale=10.0)
            + stats.t.logpdf(params["sdgp_sigma_1"], df=3.0, loc=0.0, scale=36.0) + jnp.log(2.0)
            + jnp.sum(stats.norm.logpdf(params["zgp_sigma_1"], loc=0.0, scale=1.0))
            + inv_gamma_logpdf(params["lscale_sigma_1"], a=1.124909, b=0.0177))


def mean_and_sd(*, params, datum, hyper):
    weights = gp_weights(sdgp=params["sdgp_1"], lscale=params["lscale_1"], z=params["zgp_1"],
                         slambda=hyper["slambda_1"])
    weights_sigma = gp_weights(sdgp=params["sdgp_sigma_1"], lscale=params["lscale_sigma_1"],
                               z=params["zgp_sigma_1"], slambda=hyper["slambda_sigma_1"])
    mu = params["Intercept"] + datum["Xgp_1"] @ weights
    sigma = jnp.exp(params["Intercept_sigma"] + datum["Xgp_sigma_1"] @ weights_sigma)
    return mu, sigma


def log_lik_one(params, datum, hyper):
    mu, sigma = mean_and_sd(params=params, datum=datum, hyper=hyper)
    return stats.norm.logpdf(datum["Y"], loc=mu, scale=sigma)


def sample_one(key, params, datum, hyper):
    mu, sigma = mean_and_sd(params=params, datum=datum, hyper=hyper)
    return {"Y": mu + sigma * jax.random.normal(key)}


def build(raw):
    n_basis, n_basis_sigma = int(raw["NBgp_1"]), int(raw["NBgp_sigma_1"])
    return mg.ExchangeableModel(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            Intercept=pg.NumericScalarPattern(),
            sdgp_1=pg.NumericScalarPattern(lb=0.0),
            lscale_1=pg.NumericScalarPattern(lb=0.0),
            zgp_1=pg.NumericVectorPattern(n_basis),
            Intercept_sigma=pg.NumericScalarPattern(),
            sdgp_sigma_1=pg.NumericScalarPattern(lb=0.0),
            lscale_sigma_1=pg.NumericScalarPattern(lb=0.0),
            zgp_sigma_1=pg.NumericVectorPattern(n_basis_sigma),
        ),
        prior=mg.DGP(log_density=log_prior),
        likelihood=mg.ExchangeableDGP(log_density_one=log_lik_one, sample_one=sample_one),
        data={"Y": raw["Y"], "Xgp_1": raw["Xgp_1"], "Xgp_sigma_1": raw["Xgp_sigma_1"]},
        hyper={"slambda_1": raw["slambda_1"][:, 0], "slambda_sigma_1": raw["slambda_sigma_1"][:, 0]},
    )
