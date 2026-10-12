"""Checks every model should pass, and a comparison between two models.

`check_model_contract(model)` exercises the surface methods rely on: folding
round trips, agreement between the folded/flat/free representations, the
exchangeable extras, sampling, and jit. Run it on every model in a catalog.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def check_model_contract(model, key=None, n_draws: int = 3, rtol: float = 1e-8, atol: float = 1e-8):
    key = jax.random.PRNGKey(0) if key is None else key
    pattern = model.params_pattern
    joint_free = jax.jit(model.log_density_fn("joint", params_as="free"))
    joint_flat = model.log_density_fn("joint", params_as="flat")
    joint_folded = model.log_density_fn("joint", params_as="folded")
    lik_free = model.log_density_fn("lik", params_as="free")
    for k in jax.random.split(key, n_draws):
        x = 0.5 * jax.random.normal(k, (model.free_dim,))
        params, logdet = pattern.free_to_folded_with_log_det(x)
        model.validate_params(params)
        np.testing.assert_allclose(model.flatten_params(params, to_free=True), x, rtol=1e-6, atol=1e-8)
        y = model.flatten_params(params, to_free=False)
        lp = joint_folded(params)
        _close(joint_flat(y), lp, rtol=rtol, atol=atol, what="flat vs folded joint")
        _close(joint_free(x), lp + logdet, rtol=rtol, atol=atol, what="free joint vs folded joint + log det")
        g = jax.grad(joint_free)(x)
        assert g.shape == (model.free_dim,) and np.all(np.isfinite(g)), "non-finite gradient"
        if model.exchangeable:
            pw = model.log_lik_pointwise(params)
            assert pw.shape == (model.n_obs,)
            _close(jnp.sum(pw), lik_free(x), rtol=rtol, atol=atol, what="sum of pointwise vs lik")
            _close(lik_free(x, weights=jnp.ones(model.n_obs)), lik_free(x), rtol=rtol, atol=atol, what="unit weights")
            idx = jnp.arange(model.n_obs)
            _close(model.log_lik_subset(params, idx), lik_free(x), rtol=rtol, atol=atol, what="full subset")
            gw = jax.grad(lambda w: lik_free(x, weights=w))(jnp.ones(model.n_obs))
            _close(gw, pw, rtol=rtol, atol=atol, what="d lik / d weights == pointwise")
        if model.data_pattern is not None:
            d_flat = model.data_pattern.flatten(
                {k: model.data[k] for k in model.data_pattern.keys()}, to_free=False)
            _close(model.log_density_fn("joint", params_as="free", data_as="flat")(x, d_flat), joint_free(x),
                   rtol=rtol, atol=atol, what="flat data")
    if model.capabilities["can_sample_prior"]:
        p = model.sample_prior(key)
        model.validate_params(p)
    if model.capabilities["can_sample_data"]:
        p = model.random_params(key)
        d = model.sample_data(key, p)
        model.validate_data(d)
    return True


def assert_models_match(a, b, n_draws: int = 10, key=None, rtol: float = 1e-8, atol: float = 1e-8,
                        which: str = "joint", check_grad: bool = True):
    """Two models with the same free parameterization give the same log
    density (and gradient) in free space."""
    assert a.free_dim == b.free_dim, f"free_dim {a.free_dim} != {b.free_dim}"
    key = jax.random.PRNGKey(0) if key is None else key
    fa, fb = a.log_density_fn(which, params_as="free"), b.log_density_fn(which, params_as="free")
    for k in jax.random.split(key, n_draws):
        x = jax.random.normal(k, (a.free_dim,))
        _close(fa(x), fb(x), rtol=rtol, atol=atol, what="log density")
        if check_grad:
            _close(jax.grad(fa)(x), jax.grad(fb)(x), rtol=rtol, atol=atol, what="gradient")


def _close(a, b, *, rtol, atol, what):
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol, err_msg=what)
