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
    free_length = stan_model.params_pattern.get_free_length()
    assert free_length == jax_model.params_pattern.get_free_length()
    assert_models_match(jax_model, stan_model, rtol=1e-10, atol=1e-10)
    f = jax.jit(lambda u: stan_model.eval_free_log_joint(u, jacobian=False))
    g = lambda u: jax_model.eval_log_joint(jax_model.params_pattern.free_to_folded(u))
    x = jnp.linspace(-1, 1, free_length)
    np.testing.assert_allclose(f(x), g(x), rtol=1e-10)
    np.testing.assert_allclose(jax.grad(f)(x), jax.grad(g)(x), rtol=1e-9)


def test_hessian_and_vmap(pair):
    jax_model, stan_model, _ = pair
    f, g = stan_model.eval_free_log_joint, jax_model.eval_free_log_joint
    free_length = stan_model.params_pattern.get_free_length()
    x = 0.3 * jnp.ones(free_length)
    np.testing.assert_allclose(jax.hessian(f)(x), jax.hessian(g)(x), rtol=1e-5, atol=1e-5)
    v = jnp.arange(free_length, dtype=float)
    np.testing.assert_allclose(mg.make_hvp(f)(x, v), mg.make_hvp(g)(x, v), rtol=1e-5, atol=1e-5)
    xs = jax.random.normal(jax.random.PRNGKey(0), (4, free_length))
    np.testing.assert_allclose(jax.vmap(f)(xs), jax.vmap(g)(xs), rtol=1e-10)


def test_fold_flatten_and_new_data(pair):
    jax_model, stan_model, raw = pair
    x = jax.random.normal(jax.random.PRNGKey(1), (stan_model.params_pattern.get_free_length(),))
    p = stan_model.params_pattern.free_to_folded(x)
    q = jax_model.params_pattern.free_to_folded(x)
    for k in q:
        np.testing.assert_allclose(p[k], q[k], rtol=1e-12)
    np.testing.assert_allclose(stan_model.params_pattern.folded_to_free(p), x, rtol=1e-10)
    new = {**raw, "y": [v + 1.0 for v in raw["y"]]}
    stan_new = mg.from_bridgestan(stan_model.bridgestan.lib_path, new)
    jax_new_data = {"y": jnp.asarray(new["y"]), "sigma": jnp.asarray(raw["sigma"])}
    np.testing.assert_allclose(stan_new.eval_free_log_joint(x), jax_model.eval_free_log_joint(x, data=jax_new_data),
                               rtol=1e-10)
    with pytest.raises(NotImplementedError):
        stan_model.eval_log_lik(p)
