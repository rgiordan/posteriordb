"""Shared code for nn_rbm1bJ10 and nn_rbm1bJ100: a one-hidden-layer tanh
network for MNIST digits with hierarchical (inverse-gamma) weight scales.

    transformed data {
      nu_alpha = 0.5; s2_0_alpha = (0.05 / M^(1 / nu_alpha))^2;
      nu_beta = 0.5;  s2_0_beta = (0.05 / J^(1 / nu_beta))^2;
    }
    parameters {
      real<lower=0> sigma2_alpha; real<lower=0> sigma2_beta;
      matrix[M, J] alpha; matrix[J, K - 1] beta; row_vector[J] alpha1; row_vector[K - 1] beta1;
    }
    model {
      v = append_col(ones, append_col(ones, tanh(x1 * append_row(alpha1, alpha))) * append_row(beta1, beta));
      alpha1 ~ normal(0, 1); beta1 ~ normal(0, 1);
      sigma2_alpha ~ inv_gamma(nu_alpha / 2, nu_alpha * s2_0_alpha / 2);
      sigma2_beta ~ inv_gamma(nu_beta / 2, nu_beta * s2_0_beta / 2);
      to_vector(alpha) ~ normal(0, sqrt(sigma2_alpha));
      to_vector(beta) ~ normal(0, sqrt(sigma2_beta));
      y[n] ~ categorical_logit(v[n]');
    }

Exchangeable: each image is one datum. The matrices are column-major in
Stan's free vector, hence order="F".
"""

import jax
import jax.numpy as jnp
from jax.scipy import stats
from jax.scipy.special import logsumexp

import modelgami as mg
import paragami as pg

from ._distributions import inv_gamma_logpdf

NU = 0.5


def log_prior(params, hyper):
    return (jnp.sum(stats.norm.logpdf(params["alpha1"], loc=0.0, scale=1.0))
            + jnp.sum(stats.norm.logpdf(params["beta1"], loc=0.0, scale=1.0))
            + inv_gamma_logpdf(params["sigma2_alpha"], a=NU / 2.0, b=NU * hyper["s2_0_alpha"] / 2.0)
            + inv_gamma_logpdf(params["sigma2_beta"], a=NU / 2.0, b=NU * hyper["s2_0_beta"] / 2.0)
            + jnp.sum(stats.norm.logpdf(params["alpha"], loc=0.0, scale=jnp.sqrt(params["sigma2_alpha"])))
            + jnp.sum(stats.norm.logpdf(params["beta"], loc=0.0, scale=jnp.sqrt(params["sigma2_beta"]))))


def logits(params, x):
    hidden = jnp.tanh(params["alpha1"] + x @ params["alpha"])
    return jnp.concatenate([jnp.ones(1), params["beta1"] + hidden @ params["beta"]])


def log_lik_one(params, datum, hyper):
    v = logits(params=params, x=datum["x"])
    return v[datum["y"] - 1] - logsumexp(v)


def sample_one(key, params, datum, hyper):
    draw = jax.random.categorical(key, logits=logits(params=params, x=datum["x"])) + 1
    return {"y": draw.astype(datum["y"].dtype)}


def nn_rbm_model(*, name, raw, n_hidden):
    m, k = int(raw["M"]), int(raw["K"])
    return mg.ExchangeableModel(
        name=name,
        params_pattern=pg.PatternDict(
            sigma2_alpha=pg.NumericScalarPattern(lb=0.0),
            sigma2_beta=pg.NumericScalarPattern(lb=0.0),
            alpha=pg.NumericArrayPattern((m, n_hidden), order="F"),
            beta=pg.NumericArrayPattern((n_hidden, k - 1), order="F"),
            alpha1=pg.NumericVectorPattern(n_hidden),
            beta1=pg.NumericVectorPattern(k - 1),
        ),
        prior=mg.DGP(log_density=log_prior),
        likelihood=mg.ExchangeableDGP(log_density_one=log_lik_one, sample_one=sample_one),
        data={"x": raw["x"], "y": jnp.asarray(raw["y"], dtype=jnp.int32)},
        hyper={"s2_0_alpha": (0.05 / m ** (1.0 / NU)) ** 2, "s2_0_beta": (0.05 / n_hidden ** (1.0 / NU)) ** 2},
    )
