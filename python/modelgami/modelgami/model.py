"""`Model` and `ExchangeableModel`.

A model is a prior and a likelihood (both `DGP`s), a paragami pattern for
the parameters, the data and the hyperparameters:

    model = Model(prior=DGP(eval_log_prior, draw_prior),
                  likelihood=DGP(eval_log_lik, draw_data),
                  data=data, hyper=hyper, params_pattern=pattern)

What the model itself does is combine the two: the log joint density and
draws from the joint and marginal distributions. Every method takes the data
and hyperparameters as arguments, with the model's own as defaults.

For an unconstrained target (for HMC, VI, optimization), `eval_free_log_joint`
uses the prior over the free parameters, `free_dgp(prior, params_pattern)`,
which includes the log Jacobian; the likelihood is a function of the
parameters, not a density over them, so it has none.

`ExchangeableModel` takes an `ExchangeableDGP` likelihood and adds what only
makes sense for exchangeable data: the number of observations,
single-observation and weighted log likelihoods, subsets, and bootstrap
weights.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import jax
import paragami as pg

from .dgp import DGP, ExchangeableDGP, count_obs, draw_np_bootstrap_weights, free_dgp


class Model:
    def __init__(self, *, likelihood: DGP, prior: Optional[DGP] = None, data: Any = None,
                 hyper: Optional[Dict[str, Any]] = None, params_pattern: Optional[pg.Pattern] = None,
                 name: Optional[str] = None):
        """`prior=None` is a flat (improper) prior: log density zero, no
        sampler. `params_pattern=None` infers an unconstrained pattern from
        one prior draw."""
        if not isinstance(likelihood, DGP):
            raise TypeError("likelihood must be a DGP")
        if prior is not None and not isinstance(prior, DGP):
            raise TypeError("prior must be a DGP")
        self.likelihood = likelihood
        self.prior = DGP(eval_log_density=_eval_flat_log_density) if prior is None else prior
        self.data = data
        self.hyper = {} if hyper is None else hyper
        self.name = name or type(self).__name__
        if params_pattern is None:
            try:
                example = self.draw_prior(jax.random.PRNGKey(0))
            except NotImplementedError:
                raise ValueError("give params_pattern, or a prior that can draw (to infer the pattern from)")
            params_pattern = pg.unconstrained_pattern_like(example, default_name="params")
        self.params_pattern = params_pattern

    def _data(self, data):
        return self.data if data is None else data

    def _hyper(self, hyper):
        return self.hyper if hyper is None else hyper

    # -- log densities, with the model's data and hyper as defaults ----------------

    def eval_log_prior(self, params, hyper=None):
        return self.prior.eval_log_density(params, self._hyper(hyper))

    def eval_log_lik(self, params, data=None, hyper=None, **kwargs):
        return self.likelihood.eval_log_density(self._data(data), params, self._hyper(hyper), **kwargs)

    def eval_log_joint(self, params, data=None, hyper=None, **kwargs):
        """Log prior plus log likelihood; `kwargs` go to the likelihood."""
        return self.eval_log_prior(params, hyper) + self.eval_log_lik(params, data, hyper, **kwargs)

    def eval_free_log_joint(self, free_params, data=None, hyper=None, **kwargs):
        """The log joint density as a density over the free parameters: the
        free prior (with its log Jacobian) plus the log likelihood."""
        hyper = self._hyper(hyper)
        free_prior = free_dgp(self.prior, self.params_pattern)
        params = self.params_pattern.free_to_folded(free_params)
        return free_prior.eval_log_density(free_params, hyper) + self.eval_log_lik(params, data, hyper, **kwargs)

    # -- draws ----------------------------------------------------------------------

    def draw_prior(self, key, hyper=None):
        return self.prior.draw(key, self._hyper(hyper))

    def draw_data(self, key, params, data=None, hyper=None):
        """New data given `params`, with the covariates and shapes of `data`."""
        return self.likelihood.draw(key, self._data(data), params, self._hyper(hyper))

    def draw_joint(self, key, data=None, hyper=None):
        """`(params, data)` from the joint distribution: params from the
        prior, then data given them."""
        prior_key, data_key = jax.random.split(key)
        params = self.draw_prior(prior_key, hyper)
        return params, self.draw_data(data_key, params, data, hyper)

    def draw_marginal_data(self, key, data=None, hyper=None):
        """Data from their marginal (prior predictive) distribution."""
        return self.draw_joint(key, data, hyper)[1]

    def __repr__(self):
        return f"<{type(self).__name__} {self.name}: free dimension {self.params_pattern.free_length()}>"


class ExchangeableModel(Model):
    """A model whose likelihood is an `ExchangeableDGP`: the data are N
    observations (every data leaf has a leading axis of length N),
    independent given the parameters and hyperparameters."""

    def __init__(self, *, likelihood: ExchangeableDGP, prior: Optional[DGP] = None, data: Any = None,
                 hyper: Optional[Dict[str, Any]] = None, params_pattern: Optional[pg.Pattern] = None,
                 name: Optional[str] = None):
        if not isinstance(likelihood, ExchangeableDGP):
            raise TypeError("an ExchangeableModel needs an ExchangeableDGP likelihood")
        super().__init__(likelihood=likelihood, prior=prior, data=data, hyper=hyper,
                         params_pattern=params_pattern, name=name)

    @property
    def n_obs(self) -> int:
        return count_obs(self.data)

    def eval_log_lik_singleobs(self, params, data=None, hyper=None):
        """The (N,) vector of single-observation log likelihoods."""
        return self.likelihood.eval_log_density_singleobs(self._data(data), params, self._hyper(hyper))

    def eval_log_joint(self, params, data=None, hyper=None, weights=None):
        """Log prior plus the log likelihood, weighted by `weights` if given."""
        return self.eval_log_prior(params, hyper) + self.eval_log_lik(params, data, hyper, weights=weights)

    def eval_log_joint_subset(self, params, idx, data=None, hyper=None, scale: Optional[float] = None):
        """Log prior plus the log likelihood of observations `idx`, scaled
        by N / len(idx) by default (an unbiased minibatch estimate)."""
        return self.eval_log_prior(params, hyper) + self.likelihood.eval_log_density_subset(
            self._data(data), params, self._hyper(hyper), idx, scale=scale)

    def draw_np_bootstrap_weights(self, key):
        """Nonparametric bootstrap weights for this model's observations."""
        return draw_np_bootstrap_weights(key, self.n_obs)


def make_hvp(f):
    """Hessian-vector product of a scalar function of its first argument:
    `make_hvp(f)(x, v, *rest)`, forward-over-reverse (never forms the
    Hessian)."""

    def hvp(x, v, *rest):
        return jax.jvp(lambda x_: jax.grad(f)(x_, *rest), (x,), (v,))[1]

    return hvp


def _eval_flat_log_density(params, hyper):
    return 0.0
