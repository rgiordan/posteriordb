"""One NumPyro program gives the same model as the hand-written hooks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

numpyro = pytest.importorskip("numpyro")
import numpyro.distributions as dist  # noqa: E402

import modelgami as mg  # noqa: E402
from modelgami.examples import conjugate_regression as cr  # noqa: E402
from modelgami.testing import assert_models_match, check_model_contract  # noqa: E402

jax.config.update("jax_enable_x64", True)


def regression(data, hyper):
    sigma2 = numpyro.sample("sigma2", dist.InverseGamma(hyper["a0"], hyper["b0"]))
    beta = numpyro.sample("beta", dist.MultivariateNormal(hyper["m0"], sigma2 * hyper["V0"]))
    with numpyro.plate("obs", data["x"].shape[0]):
        numpyro.sample("y", dist.Normal(data["x"] @ beta, jnp.sqrt(sigma2)), obs=data["y"])


@pytest.fixture(scope="module")
def models():
    X, y = cr.simulate_data(jax.random.PRNGKey(0), n=40, p=3)
    hand = cr.make_model(X, y)
    npm = mg.from_numpyro(regression, data={"x": X, "y": y}, hyper=hand.hyper, exchangeable=True)
    return hand, npm


def test_same_model(models):
    hand, npm = models
    # NumPyro orders sites as the program runs (sigma2 first); compare in folded space.
    assert set(npm.params_pattern.keys()) == {"beta", "sigma2"}
    for k in jax.random.split(jax.random.PRNGKey(1), 5):
        p = hand.params_pattern.random(k)
        np.testing.assert_allclose(npm.eval_log_prior(p), hand.eval_log_prior(p), rtol=1e-10)
        np.testing.assert_allclose(npm.eval_log_lik_singleobs(p), hand.eval_log_lik_singleobs(p), rtol=1e-10)
        x = npm.params_pattern.folded_to_free(p)
        np.testing.assert_allclose(npm.eval_free_log_joint(x),
                                   hand.eval_free_log_joint(hand.params_pattern.folded_to_free(p)), rtol=1e-10)


def test_contract_and_simulation(models):
    hand, npm = models
    check_model_contract(npm)
    assert isinstance(npm, mg.ExchangeableModel)
    p = npm.draw_prior(jax.random.PRNGKey(2))
    assert npm.params_pattern.validate_folded(p)[0]
    d = npm.draw_data(jax.random.PRNGKey(3), p)
    np.testing.assert_array_equal(d["x"], hand.data["x"])
    assert d["y"].shape == hand.data["y"].shape and not np.allclose(d["y"], hand.data["y"])


def test_hyper_and_data_derivatives(models):
    _, npm = models
    x = jnp.zeros(npm.params_pattern.free_length())
    gh = jax.grad(lambda h: npm.eval_free_log_joint(x, hyper=h))(npm.hyper)
    assert np.all(np.isfinite(gh["V0"]))
    gd = jax.grad(lambda d: npm.eval_free_log_joint(x, data=d))(npm.data)
    assert np.all(np.isfinite(gd["x"])) and np.all(np.isfinite(gd["y"]))


def test_constraint_mapping():
    def mix(data, hyper):
        w = numpyro.sample("w", dist.Dirichlet(jnp.ones(3)))
        mu = numpyro.sample("mu", dist.Normal(jnp.zeros(3), 5.0))
        s = numpyro.sample("s", dist.HalfNormal(1.0))
        L = numpyro.sample("L", dist.LKJCholesky(2, 2.0))
        r = numpyro.sample("r", dist.Uniform(-1.0, 2.0))
        with numpyro.plate("obs", data["y"].shape[0]):
            numpyro.sample("y", dist.MixtureSameFamily(dist.Categorical(w), dist.Normal(mu, s)), obs=data["y"])

    m = mg.from_numpyro(mix, data={"y": jnp.linspace(-2, 2, 7)}, exchangeable=True)
    pat = m.params_pattern
    assert type(pat["w"]).__name__ == "SimplexArrayPattern"
    assert type(pat["L"]).__name__ == "CholeskyCorrPattern"
    assert pat["s"].lb == 0.0 and (pat["r"].lb, pat["r"].ub) == (-1.0, 2.0)
    check_model_contract(m)
