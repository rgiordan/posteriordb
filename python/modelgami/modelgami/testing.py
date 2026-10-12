"""Checks every model should pass, and a comparison between two models.

`check_model_contract(model)` checks that the folded, flat and free log
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
    joint_free = jax.jit(model.log_density_fn("joint", params_as="free"))
    joint_flat = model.log_density_fn("joint", params_as="flat")
    joint_folded = model.log_density_fn("joint", params_as="folded")
    lik_free = model.log_density_fn("lik", params_as="free")
    for k in jax.random.split(key, n_draws):
        x = 0.5 * jax.random.normal(k, (pattern.free_length(),))
        params = pattern.free_to_folded(x)
        logdet = pattern.free_to_folded_log_abs_det(x)
        np.testing.assert_allclose(pattern.folded_to_free(params), x, rtol=1e-6, atol=1e-8)
        lp = joint_folded(params)
        _close(joint_flat(pattern.folded_to_flat(params)), lp, rtol=rtol, atol=atol, what="flat vs folded joint")
        _close(joint_free(x), lp + logdet, rtol=rtol, atol=atol, what="free joint vs folded joint + log det")
        g = jax.grad(joint_free)(x)
        assert g.shape == x.shape and np.all(np.isfinite(g)), "non-finite gradient"
        if isinstance(model, ExchangeableModel):
            pw = model.log_lik_pointwise(params)
            assert pw.shape == (model.n_obs,)
            _close(jnp.sum(pw), lik_free(x), rtol=rtol, atol=atol, what="sum of pointwise vs lik")
            _close(lik_free(x, weights=jnp.ones(model.n_obs)), lik_free(x), rtol=rtol, atol=atol,
                   what="unit weights")
            gw = jax.grad(lambda w: lik_free(x, weights=w))(jnp.ones(model.n_obs))
            _close(gw, pw, rtol=rtol, atol=atol, what="d lik / d weights == pointwise")
    try:
        params = model.sample_prior(key)
        ok, msg = pattern.validate_folded(params)
        assert ok, f"prior draw: {msg}"
    except NotImplementedError:
        params = pattern.random(key)
    try:
        model.sample_data(key, params)
    except NotImplementedError:
        pass
    return True


def assert_models_match(a, b, n_draws: int = 10, key=None, rtol: float = 1e-8, atol: float = 1e-8,
                        which: str = "joint", check_grad: bool = True):
    """Two models with the same free parameterization give the same log
    density (and gradient) in free space."""
    dim = a.params_pattern.free_length()
    assert dim == b.params_pattern.free_length(), "free dimensions differ"
    key = jax.random.PRNGKey(0) if key is None else key
    fa, fb = a.log_density_fn(which, params_as="free"), b.log_density_fn(which, params_as="free")
    for k in jax.random.split(key, n_draws):
        x = jax.random.normal(k, (dim,))
        _close(fa(x), fb(x), rtol=rtol, atol=atol, what="log density")
        if check_grad:
            _close(jax.grad(fa)(x), jax.grad(fb)(x), rtol=rtol, atol=atol, what="gradient")


def _close(a, b, *, rtol, atol, what):
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol, err_msg=what)
