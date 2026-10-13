"""Shared code for the latent Dirichlet allocation models: each word token is
one datum carrying its word and document indices, and its topic is summed out."""

import jax
import jax.numpy as jnp
from jax.scipy import stats
from jax.scipy.special import logsumexp

import modelgami as mg
import paragami as pg


def log_prior(params, hyper):
    """theta[m] ~ dirichlet(alpha); phi[k] ~ dirichlet(beta)."""
    theta_lp = jax.vmap(lambda t: stats.dirichlet.logpdf(t, alpha=hyper["alpha"]))(params["theta"])
    phi_lp = jax.vmap(lambda p: stats.dirichlet.logpdf(p, alpha=hyper["beta"]))(params["phi"])
    return jnp.sum(theta_lp) + jnp.sum(phi_lp)


def log_lik_one(params, datum, hyper):
    """log_sum_exp_k(log(theta[doc, k]) + log(phi[k, w]))."""
    theta_doc = params["theta"][datum["doc"] - 1]                 # (K,)
    phi_word = params["phi"][:, datum["w"] - 1]                   # (K,)
    return logsumexp(jnp.log(theta_doc) + jnp.log(phi_word))


def lda_model(*, name, raw, n_topics, alpha, beta):
    m, v = int(raw["M"]), int(raw["V"])
    return mg.ExchangeableModel(
        name=name,
        params_pattern=pg.PatternDict(
            theta=pg.SimplexArrayPattern(n_topics, array_shape=(m,)),
            phi=pg.SimplexArrayPattern(v, array_shape=(n_topics,)),
        ),
        prior=mg.DGP(log_density=log_prior),
        likelihood=mg.ExchangeableDGP(log_density_one=log_lik_one),
        data={"w": jnp.asarray(raw["w"], dtype=jnp.int32), "doc": jnp.asarray(raw["doc"], dtype=jnp.int32)},
        hyper={"alpha": jnp.asarray(alpha, dtype=jnp.float64), "beta": jnp.asarray(beta, dtype=jnp.float64)},
    )
