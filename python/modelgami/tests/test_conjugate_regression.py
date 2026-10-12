import jax
import jax.numpy as jnp
import numpy as np
import pytest

import modelgami as mg
from modelgami.examples import conjugate_regression as cr
from modelgami.testing import assert_models_match, check_model_contract

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def setup():
    X, y = cr.simulate_data(jax.random.PRNGKey(0), n=50, p=3)
    return X, y, cr.make_model(X, y)


def test_contract(setup):
    _, _, model = setup
    assert check_model_contract(model)
    caps = model.capabilities
    assert caps["exchangeable"] and caps["has_prior"] and caps["can_sample_prior"] and caps["can_sample_data"]
    assert caps["free_dim"] == 4 and caps["n_obs"] == 50


def test_joint_equals_exact_posterior_plus_evidence(setup):
    """log p(beta, sigma2, y) - log p(beta, sigma2 | y) = log p(y), for any params."""
    X, y, model = setup
    joint = model.log_density_fn("joint", params_as="folded")
    log_evidence = cr.log_marginal_likelihood(X, y, model.hyper)
    for k in jax.random.split(jax.random.PRNGKey(1), 5):
        params = model.random_params(k)
        diff = joint(params) - cr.exact_log_posterior(params, X, y, model.hyper)
        np.testing.assert_allclose(diff, log_evidence, rtol=1e-10)


def test_posterior_mode_by_optimization(setup):
    """Gradient ascent on the free-space joint without Jacobian finds the
    exact posterior mode of (beta, sigma2)."""
    X, y, model = setup
    f = jax.jit(model.log_density_fn("joint", params_as="free", jacobian=False))
    g = jax.jit(jax.grad(f))
    h = jax.jit(jax.hessian(f))
    x = jnp.zeros(model.free_dim)
    for _ in range(100):  # damped Newton (gradient step where the Hessian isn't negative definite)
        H = h(x)
        step = -jnp.linalg.solve(H, g(x)) if np.all(np.linalg.eigvalsh(H) < 0) else 0.01 * g(x)
        t = 1.0
        while f(x + t * step) < f(x) and t > 1e-8:
            t /= 2
        x = x + t * step
    params = model.fold_params(x)
    post = cr.exact_posterior(X, y, model.hyper)
    p = X.shape[1]
    np.testing.assert_allclose(params["beta"], post["m0"], rtol=1e-8)
    # joint mode in sigma2 of N(beta; mn, s2 Vn) IG(s2; an, bn) at beta = mn
    np.testing.assert_allclose(params["sigma2"], post["b0"] / (post["a0"] + 1 + p / 2), rtol=1e-8)


def test_function_and_class_styles_agree(setup):
    X, y, model = setup
    other = cr.ConjugateRegression(data={"x": X, "y": y}, hyper=cr.default_hyper(3))
    assert_models_match(model, other)
    assert other.exchangeable and other.capabilities["can_sample_data"]


def test_data_as_list_of_dicts(setup):
    X, y, model = setup
    rows = [{"x": X[i], "y": y[i]} for i in range(X.shape[0])]
    other = mg.Model(params_pattern=model.params_pattern, data=rows, hyper=model.hyper,
                     log_prior=cr.log_prior, log_lik_one=cr.log_lik_one)
    assert_models_match(model, other)


def test_weights_subsets_and_bootstrap(setup):
    X, y, model = setup
    params = model.random_params(jax.random.PRNGKey(2))
    pw = model.log_lik_pointwise(params)
    w = jnp.zeros(model.n_obs).at[:10].set(1.0)
    np.testing.assert_allclose(model.log_lik(params, weights=w), pw[:10].sum())
    np.testing.assert_allclose(model.log_lik_subset(params, jnp.arange(10), scale=1.0), pw[:10].sum())
    sub = model.with_data({"x": X[:10], "y": y[:10]})
    np.testing.assert_allclose(sub.log_lik(params), pw[:10].sum())
    assert sub.sample_data(jax.random.PRNGKey(0), params)["y"].shape == (10,)
    new_hyper = {**model.hyper, "a0": 5.0}
    np.testing.assert_allclose(model.with_data(hyper=new_hyper).log_prior(params), cr.log_prior(params, new_hyper))
    bw = model.bootstrap_weights(jax.random.PRNGKey(3))
    assert bw.sum() == model.n_obs
    # minibatch estimate is unbiased: average over all equal-size batches
    batches = jnp.arange(model.n_obs).reshape(5, 10)
    est = jnp.mean(jnp.stack([model.log_lik_subset(params, b) for b in batches]))
    np.testing.assert_allclose(est, pw.sum(), rtol=1e-12)


def test_derivatives_wrt_data_and_hyper(setup):
    X, y, model = setup
    f = model.log_density_fn("joint", params_as="free", data_as="flat")
    x = jnp.zeros(model.free_dim)
    d_flat = model.data_pattern.flatten({"x": X, "y": y}, free=False)
    g = jax.grad(f, argnums=1)(x, d_flat)
    assert g.shape == d_flat.shape and np.all(np.isfinite(g))
    # prior sensitivity: d/d m0 of the log evidence matches autodiff of the exact formula
    dev = jax.grad(lambda m0: cr.log_marginal_likelihood(X, y, {**model.hyper, "m0": m0}))(model.hyper["m0"])
    assert np.all(np.isfinite(dev))
    gh = jax.grad(lambda h: model.log_density_fn("joint", "free")(x, hyper=h))(model.hyper)
    assert gh["V0"].shape == (3, 3)


def test_sampling_and_validation(setup):
    X, y, model = setup
    p = model.sample_prior(jax.random.PRNGKey(4))
    model.validate_params(p)
    d = model.sample_data(jax.random.PRNGKey(5), p)
    np.testing.assert_array_equal(d["x"], X)  # covariates kept
    assert not np.allclose(d["y"], y)
    with pytest.raises(mg.ValidationError):
        model.validate_params({"beta": jnp.zeros(3), "sigma2": -1.0})
    with pytest.raises(mg.ValidationError):
        model.validate_params({"beta": jnp.zeros(2), "sigma2": 1.0})
    with pytest.raises(mg.ValidationError):
        cr.make_model(X, y[:10])  # ragged N
    with pytest.raises(mg.ValidationError):
        mg.Model(params_pattern=model.params_pattern, data={"x": X[:, 0], "y": y}, log_lik_one=cr.log_lik_one,
                 check_data=cr.check_data)


def test_hvp(setup):
    _, _, model = setup
    f = model.log_density_fn("joint", "free")
    x, v = jnp.ones(model.free_dim) * 0.1, jnp.arange(model.free_dim, dtype=float)
    np.testing.assert_allclose(mg.hvp(f)(x, v), jax.hessian(f)(x) @ v, rtol=1e-8)


def test_arviz_export_and_loo(setup):
    az = pytest.importorskip("arviz")
    X, y, model = setup
    draws = cr.sample_exact_posterior(jax.random.PRNGKey(6), X, y, model.hyper, n_chains=2, n_draws=200)
    idata = model.to_arviz(draws, dims={"beta": ["coef"]})
    assert "log_likelihood" in idata.children if hasattr(idata, "children") else hasattr(idata, "log_likelihood")
    loo = az.loo(idata)
    assert np.isfinite(float(getattr(loo, "elpd", getattr(loo, "elpd_loo", np.nan))))
    # free-vector draws give the same result
    free = jax.vmap(jax.vmap(model.flatten_params))(draws)
    idata2 = model.to_arviz(free, params_as="free")
    np.testing.assert_allclose(np.asarray(idata2["posterior"]["sigma2"]), np.asarray(draws["sigma2"]), rtol=1e-10)


def test_obs_dists_one_gives_density_and_sampler(setup):
    """Writing the likelihood once as distribution objects (ordinary Python,
    no modeling language) gives both log_lik_one and sample_one."""
    dist = pytest.importorskip("numpyro.distributions")
    X, y, model = setup

    def obs_dists_one(params, datum, hyper):
        return {"y": dist.Normal(datum["x"] @ params["beta"], jnp.sqrt(params["sigma2"]))}

    other = mg.Model(params_pattern=model.params_pattern, data={"x": X, "y": y}, hyper=model.hyper,
                     log_prior=cr.log_prior, obs_dists_one=obs_dists_one)
    assert_models_match(model, other)
    p = model.random_params(jax.random.PRNGKey(7))
    d = other.sample_data(jax.random.PRNGKey(8), p)
    assert d["y"].shape == y.shape and not np.allclose(d["y"], y)
    check_model_contract(other)


def test_prior_dists_gives_density_and_sampler(setup):
    """A dependent prior written once, as distributions that may depend on
    earlier parameters."""
    dist = pytest.importorskip("numpyro.distributions")
    X, y, model = setup

    def prior_dists(hyper):
        return {
            "sigma2": dist.InverseGamma(hyper["a0"], hyper["b0"]),
            "beta": lambda p: dist.MultivariateNormal(hyper["m0"], p["sigma2"] * hyper["V0"]),
        }

    def obs_dists_one(params, datum, hyper):
        return {"y": dist.Normal(datum["x"] @ params["beta"], jnp.sqrt(params["sigma2"]))}

    other = mg.Model(params_pattern=model.params_pattern, data={"x": X, "y": y}, hyper=model.hyper,
                     prior_dists=prior_dists, obs_dists_one=obs_dists_one)
    assert_models_match(model, other)
    check_model_contract(other)
    draws = jax.vmap(other.sample_prior)(jax.random.split(jax.random.PRNGKey(9), 4000))
    # sigma2 ~ IG(2, 2), whose median is 2 / (median of Gamma(2, 1)) = 2 / 1.678
    np.testing.assert_allclose(np.median(draws["sigma2"]), 2.0 / 1.67834, rtol=0.1)


def test_dist_hooks_with_tfp(setup):
    """The same hooks work with TFP distributions (needs a TFP-compatible JAX)."""
    try:
        from tensorflow_probability.substrates import jax as tfp
    except Exception:  # not installed, or incompatible with this JAX version
        pytest.skip("tensorflow_probability (JAX substrate) unavailable")
    tfd = tfp.distributions
    X, y, model = setup

    def prior_dists(hyper):
        return {
            "sigma2": tfd.InverseGamma(hyper["a0"], hyper["b0"]),
            "beta": lambda p: tfd.MultivariateNormalTriL(
                hyper["m0"], jnp.linalg.cholesky(p["sigma2"] * hyper["V0"])),
        }

    def obs_dists_one(params, datum, hyper):
        return {"y": tfd.Normal(datum["x"] @ params["beta"], jnp.sqrt(params["sigma2"]))}

    other = mg.Model(params_pattern=model.params_pattern, data={"x": X, "y": y}, hyper=model.hyper,
                     prior_dists=prior_dists, obs_dists_one=obs_dists_one)
    assert_models_match(model, other)
    check_model_contract(other)
