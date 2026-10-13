"""Checks every model should pass, and a comparison between two models.

`check_model_contract(model)` checks that the free and folded log joint
densities agree, that gradients are finite, that the exchangeable extras
agree with the likelihood, and that whatever samplers exist run. Run it on
every model in a catalog.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .model import ExchangeableModel


def check_model_contract(model, key=None, n_draws: int = 3, rtol: float = 1e-8, atol: float = 1e-8):
    key = jax.random.PRNGKey(0) if key is None else key
    pattern = model.params_pattern
    eval_free = jax.jit(model.eval_free_log_joint)
    for k in jax.random.split(key, n_draws):
        x = 0.5 * jax.random.normal(k, (pattern.get_free_length(),))
        params, log_det = pattern.free_to_folded_with_log_det(x)
        np.testing.assert_allclose(pattern.folded_to_free(params), x, rtol=1e-6, atol=1e-8)
        _close(eval_free(x), model.eval_log_joint(params) + log_det, rtol=rtol, atol=atol,
               what="free log joint vs folded log joint + log det")
        g = jax.grad(eval_free)(x)
        assert g.shape == x.shape and np.all(np.isfinite(g)), "non-finite gradient"
        if isinstance(model, ExchangeableModel):
            singleobs = model.eval_log_lik_singleobs(params)
            assert singleobs.shape == (model.n_obs,)
            _close(jnp.sum(singleobs), model.eval_log_lik(params), rtol=rtol, atol=atol,
                   what="sum of single observations vs log lik")
            ones = jnp.ones(model.n_obs)
            _close(model.eval_log_joint(params, weights=ones), model.eval_log_joint(params), rtol=rtol, atol=atol,
                   what="unit weights")
            gw = jax.grad(lambda w: model.eval_log_joint(params, weights=w))(ones)
            _close(gw, singleobs, rtol=rtol, atol=atol, what="d log joint / d weights == single observations")
    try:
        params = model.draw_prior(key)
        ok, msg = pattern.validate_folded(params)
        assert ok, f"prior draw: {msg}"
    except NotImplementedError:
        params = pattern.draw_random(key)
    try:
        model.draw_data(key, params)
    except NotImplementedError:
        pass
    return True


def assert_models_match(a, b, n_draws: int = 10, key=None, rtol: float = 1e-8, atol: float = 1e-8,
                        check_grad: bool = True):
    """Two models with the same free parameterization give the same free
    log joint density (and gradient)."""
    dim = a.params_pattern.get_free_length()
    assert dim == b.params_pattern.get_free_length(), "free dimensions differ"
    key = jax.random.PRNGKey(0) if key is None else key
    for k in jax.random.split(key, n_draws):
        x = jax.random.normal(k, (dim,))
        _close(a.eval_free_log_joint(x), b.eval_free_log_joint(x), rtol=rtol, atol=atol, what="log density")
        if check_grad:
            _close(jax.grad(a.eval_free_log_joint)(x), jax.grad(b.eval_free_log_joint)(x), rtol=rtol, atol=atol,
                   what="gradient")


def _close(a, b, *, rtol, atol, what):
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol, err_msg=what)
