"""logistic_regression_rhs.stan (logistic regression with a regularized
horseshoe prior):

    parameters { real beta0; vector[d] z; real<lower=0> tau; vector<lower=0>[d] lambda; real<lower=0> caux; }
    transformed parameters {
      c = slab_scale * sqrt(caux);
      lambda_tilde = sqrt(c^2 * lambda^2 ./ (c^2 + tau^2 * lambda^2));
      beta = z .* lambda_tilde * tau;
    }
    model {
      z ~ std_normal();
      lambda ~ student_t(nu_local, 0, 1);
      tau ~ student_t(nu_global, 0, scale_global * 2);
      caux ~ inv_gamma(0.5 * slab_df, 0.5 * slab_df);
      beta0 ~ normal(0, scale_icept);
      y ~ bernoulli_logit_glm(x, beta0, beta);
    }

The prior scales and degrees of freedom are hyperparameters.
"""

import jax
import jax.numpy as jnp
from jax.scipy import stats

import modelgami as mg
import paragami as pg

from ._distributions import eval_inv_gamma_logpdf

STAN_MODEL = "logistic_regression_rhs"
POSTERIORS = ["ovarian-logistic_regression_rhs", "prostate-logistic_regression_rhs"]

HYPER_NAMES = ["scale_icept", "scale_global", "nu_global", "nu_local", "slab_scale", "slab_df"]


def eval_log_prior(params, hyper):
    return (jnp.sum(stats.norm.logpdf(params["z"], loc=0.0, scale=1.0))
            + jnp.sum(stats.t.logpdf(params["lambda"], df=hyper["nu_local"], loc=0.0, scale=1.0))
            + stats.t.logpdf(params["tau"], df=hyper["nu_global"], loc=0.0, scale=2.0 * hyper["scale_global"])
            + eval_inv_gamma_logpdf(params["caux"], a=0.5 * hyper["slab_df"], b=0.5 * hyper["slab_df"])
            + stats.norm.logpdf(params["beta0"], loc=0.0, scale=hyper["scale_icept"]))


def compute_beta(params, hyper):
    """The transformed parameter beta (regularized horseshoe coefficients)."""
    c2 = hyper["slab_scale"] ** 2 * params["caux"]
    lambda2 = params["lambda"] ** 2
    lambda_tilde = jnp.sqrt(c2 * lambda2 / (c2 + params["tau"] ** 2 * lambda2))
    return params["z"] * lambda_tilde * params["tau"]


def eval_log_lik_one(datum, params, hyper):
    eta = params["beta0"] + datum["x"] @ compute_beta(params=params, hyper=hyper)
    return datum["y"] * eta - jax.nn.softplus(eta)


def draw_one(key, datum, params, hyper):
    p = jax.nn.sigmoid(params["beta0"] + datum["x"] @ compute_beta(params=params, hyper=hyper))
    return {"y": jax.random.bernoulli(key, p=p).astype(datum["y"].dtype)}


def build(raw):
    d = int(raw["d"])
    return mg.ExchangeableModel(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            beta0=pg.NumericScalarPattern(),
            z=pg.NumericVectorPattern(d),
            tau=pg.NumericScalarPattern(lb=0.0),
            # `lambda` is a Python keyword, so this entry is passed as a dict.
            **{"lambda": pg.NumericVectorPattern(d, lb=0.0)},
            caux=pg.NumericScalarPattern(lb=0.0),
        ),
        prior=mg.DGP(eval_log_density=eval_log_prior),
        likelihood=mg.ExchangeableDGP(eval_log_density_one=eval_log_lik_one, draw_one=draw_one),
        data={"y": jnp.asarray(raw["y"], dtype=jnp.int32), "x": raw["x"]},
        hyper={k: float(raw[k]) for k in HYPER_NAMES},
    )
