"""The basketball drive hidden Markov models (hmm_drive_0, hmm_drive_1): two
states, two transition rows with Dirichlet priors, ordered emission
parameters phi (for 1/speed u) and lambda (for hoop distance v)."""

import jax.numpy as jnp
from jax.scipy import stats

import modelgami as mg
import paragami as pg

from ._hmm import forward_log_marginal


def log_prior(params, hyper):
    alpha = hyper["alpha"]
    return (stats.dirichlet.logpdf(params["theta1"], alpha[0])
            + stats.dirichlet.logpdf(params["theta2"], alpha[1])
            + stats.norm.logpdf(params["phi"][0], loc=0.0, scale=1.0)
            + stats.norm.logpdf(params["phi"][1], loc=3.0, scale=1.0)
            + stats.norm.logpdf(params["lambda"][0], loc=0.0, scale=1.0)
            + stats.norm.logpdf(params["lambda"][1], loc=3.0, scale=1.0))


def log_lik_from_emission(*, params, log_emission):
    log_transition = jnp.log(jnp.stack([params["theta1"], params["theta2"]]))   # [from, to]
    return forward_log_marginal(log_transition=log_transition, log_emission=log_emission)


def drive_model(*, name, raw, log_lik, positive, hyper):
    k = int(raw["K"])
    return mg.Model(
        name=name,
        params_pattern=pg.PatternDict(
            theta1=pg.SimplexArrayPattern(k),
            theta2=pg.SimplexArrayPattern(k),
            phi=pg.OrderedVectorPattern(k, positive=positive),
            **{"lambda": pg.OrderedVectorPattern(k, positive=positive)},
        ),
        prior=mg.DGP(log_density=log_prior),
        likelihood=mg.DGP(log_density=log_lik),
        data={"u": raw["u"], "v": raw["v"]},
        hyper={"alpha": jnp.asarray(raw["alpha"]), **hyper},
    )
