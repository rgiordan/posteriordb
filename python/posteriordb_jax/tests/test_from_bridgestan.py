"""modelgami.from_bridgestan: a Stan program used directly as a Model,
compared with its JAX translation from the catalog."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import modelgami as mg
import posteriordb_jax  # noqa: F401
from modelgami.testing import assert_models_match
from posteriordb_jax import catalog
from posteriordb_jax.io import load_data, stan_file_path

pytestmark = pytest.mark.stan


@pytest.fixture(scope="module")
def pair():
    raw = load_data("eight_schools")
    return catalog.load("eight_schools-eight_schools_noncentered"), mg.from_bridgestan(
        stan_file_path("eight_schools_noncentered"), raw), raw


def test_density_and_gradient_match(pair):
    jax_model, stan_model, _ = pair
    assert stan_model.free_dim == jax_model.free_dim
    assert_models_match(jax_model, stan_model, rtol=1e-10, atol=1e-10)
    f = jax.jit(stan_model.log_density_fn("joint", params_as="free", jacobian=False))
    g = jax_model.log_density_fn("joint", params_as="free", jacobian=False)
    x = jnp.linspace(-1, 1, stan_model.free_dim)
    np.testing.assert_allclose(f(x), g(x), rtol=1e-10)
    np.testing.assert_allclose(jax.grad(f)(x), jax.grad(g)(x), rtol=1e-9)


def test_hessian_and_vmap(pair):
    jax_model, stan_model, _ = pair
    f, g = stan_model.log_density_fn(), jax_model.log_density_fn()
    x = 0.3 * jnp.ones(stan_model.free_dim)
    np.testing.assert_allclose(jax.hessian(f)(x), jax.hessian(g)(x), rtol=1e-5, atol=1e-5)
    v = jnp.arange(stan_model.free_dim, dtype=float)
    np.testing.assert_allclose(mg.hvp(f)(x, v), mg.hvp(g)(x, v), rtol=1e-5, atol=1e-5)
    xs = jax.random.normal(jax.random.PRNGKey(0), (4, stan_model.free_dim))
    np.testing.assert_allclose(jax.vmap(f)(xs), jax.vmap(g)(xs), rtol=1e-10)


def test_fold_flatten_and_with_data(pair):
    jax_model, stan_model, raw = pair
    x = jax.random.normal(jax.random.PRNGKey(1), (stan_model.free_dim,))
    p = stan_model.fold_params(x, from_free=True)
    q = jax_model.fold_params(x, from_free=True)
    for k in q:
        np.testing.assert_allclose(p[k], q[k], rtol=1e-12)
    np.testing.assert_allclose(stan_model.flatten_params(p, to_free=True), x, rtol=1e-10)
    new = {**raw, "y": [v + 1.0 for v in raw["y"]]}
    s2, j2 = stan_model.with_data(new), jax_model.with_data({"y": new["y"], "sigma": raw["sigma"]})
    np.testing.assert_allclose(s2.log_density_fn()(x), j2.log_density_fn()(x), rtol=1e-10)
    with pytest.raises(NotImplementedError):
        stan_model.log_density_fn("lik")
