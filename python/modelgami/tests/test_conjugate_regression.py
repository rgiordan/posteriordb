import jax
import jax.numpy as jnp
import numpy as np
import pytest

try:
    from tensorflow_probability.substrates import jax as tfp
except Exception:  # not installed, or incompatible with this JAX version
    tfp = None
try:
    import numpyro.distributions as npdist
except ImportError:
    npdist = None

import modelgami as mg
import paragami as pg
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
    assert model.n_obs == 50 and model.params_pattern.free_length() == 4


def test_joint_equals_exact_posterior_plus_evidence(setup):
    """log p(beta, sigma2, y) - log p(beta, sigma2 | y) = log p(y), for any params."""
    X, y, model = setup
    joint = model.log_density_fn("joint", params_as="folded")
    log_evidence = cr.log_marginal_likelihood(X, y, model.hyper)
    for k in jax.random.split(jax.random.PRNGKey(1), 5):
        params = model.params_pattern.random(k)
        diff = joint(params) - cr.exact_log_posterior(params, X, y, model.hyper)
        np.testing.assert_allclose(diff, log_evidence, rtol=1e-10)


def test_posterior_mode_by_optimization(setup):
    """Newton's method on the free-space joint without Jacobian finds the
    exact posterior mode of (beta, sigma2)."""
    X, y, model = setup
    f = jax.jit(model.log_density_fn("joint", params_as="free", jacobian=False))
    g = jax.jit(jax.grad(f))
    h = jax.jit(jax.hessian(f))
    x = jnp.zeros(model.params_pattern.free_length())
    for _ in range(100):  # damped Newton (gradient step where the Hessian isn't negative definite)
        H = h(x)
        step = -jnp.linalg.solve(H, g(x)) if np.all(np.linalg.eigvalsh(H) < 0) else 0.01 * g(x)
        t = 1.0
        while f(x + t * step) < f(x) and t > 1e-8:
            t /= 2
        x = x + t * step
    params = model.params_pattern.free_to_folded(x)
    post = cr.exact_posterior(X, y, model.hyper)
    p = X.shape[1]
    np.testing.assert_allclose(params["beta"], post["m0"], rtol=1e-8)
    # joint mode in sigma2 of N(beta; mn, s2 Vn) IG(s2; an, bn) at beta = mn
    np.testing.assert_allclose(params["sigma2"], post["b0"] / (post["a0"] + 1 + p / 2), rtol=1e-8)


def test_function_and_class_styles_agree(setup):
    X, y, model = setup
    other = mg.ExchangeableModel(params_pattern=cr.params_pattern(3), prior=cr.NormalInverseGammaPrior(),
                                 likelihood=cr.NormalRegressionLikelihood(), data={"x": X, "y": y},
                                 hyper=cr.default_hyper(3))
    assert_models_match(model, other)
    check_model_contract(other)


def test_general_model_matches_exchangeable(setup):
    """The same likelihood written for the whole data set, in a plain Model."""
    X, y, model = setup

    def log_lik(params, data, hyper):
        return jnp.sum(jax.vmap(cr.log_lik_one, in_axes=(None, 0, None))(params, data, hyper))

    other = mg.Model(params_pattern=model.params_pattern, prior=model.prior, likelihood=mg.DGP(log_lik),
                     data=model.data, hyper=model.hyper)
    assert_models_match(model, other)
    with pytest.raises(NotImplementedError):
        other.sample_data(jax.random.PRNGKey(0), model.params_pattern.random(jax.random.PRNGKey(1)))


def test_missing_pieces_raise(setup):
    X, y, model = setup
    no_prior = mg.ExchangeableModel(params_pattern=model.params_pattern,
                                    likelihood=mg.ExchangeableDGP(cr.log_lik_one), data=model.data)
    params = model.params_pattern.random(jax.random.PRNGKey(0))
    assert no_prior.log_prior(params) == 0.0  # flat prior
    with pytest.raises(NotImplementedError):
        no_prior.sample_prior(jax.random.PRNGKey(0))
    with pytest.raises(NotImplementedError):
        no_prior.sample_data(jax.random.PRNGKey(0), params)
    with pytest.raises(NotImplementedError):
        mg.DGP(sample=cr.sample_prior).log_density(params, model.hyper)
    with pytest.raises(TypeError):
        mg.ExchangeableModel(params_pattern=model.params_pattern, likelihood=mg.DGP(), data=model.data)


def test_weights_subsets_and_bootstrap(setup):
    X, y, model = setup
    params = model.params_pattern.random(jax.random.PRNGKey(2))
    pw = model.log_lik_pointwise(params)
    w = jnp.zeros(model.n_obs).at[:10].set(1.0)
    np.testing.assert_allclose(model.log_lik(params, weights=w), pw[:10].sum())
    lik = model.likelihood
    np.testing.assert_allclose(lik.log_density_subset(params, model.data, model.hyper, jnp.arange(10), scale=1.0),
                               pw[:10].sum())
    small = {"x": X[:10], "y": y[:10]}
    np.testing.assert_allclose(model.log_lik(params, data=small), pw[:10].sum())
    assert model.sample_data(jax.random.PRNGKey(0), params, data=small)["y"].shape == (10,)
    new_hyper = {**model.hyper, "a0": 5.0}
    np.testing.assert_allclose(model.log_prior(params, hyper=new_hyper), cr.log_prior(params, new_hyper))
    bw = model.bootstrap_weights(jax.random.PRNGKey(3))
    assert bw.sum() == model.n_obs
    # minibatch estimate is unbiased: average over all equal-size batches
    batches = jnp.arange(model.n_obs).reshape(5, 10)
    est = jnp.mean(jnp.stack([lik.log_density_subset(params, model.data, model.hyper, b) for b in batches]))
    np.testing.assert_allclose(est, pw.sum(), rtol=1e-12)


def test_derivatives_wrt_data_and_hyper(setup):
    X, y, model = setup
    f = model.log_density_fn("joint", params_as="free")
    x = jnp.zeros(model.params_pattern.free_length())
    # Derivatives with respect to data, through a paragami pattern for one datum.
    datum_pattern = pg.PatternDict(x=pg.NumericVectorPattern(3), y=pg.NumericScalarPattern())
    flat = jax.vmap(datum_pattern.folded_to_flat)(model.data)
    g = jax.grad(lambda d: f(x, data=jax.vmap(datum_pattern.flat_to_folded)(d)))(flat)
    assert g.shape == flat.shape and np.all(np.isfinite(g))
    gh = jax.grad(lambda h: f(x, hyper=h))(model.hyper)
    assert gh["V0"].shape == (3, 3)


def test_sampling(setup):
    X, y, model = setup
    p = model.sample_prior(jax.random.PRNGKey(4))
    assert model.params_pattern.validate_folded(p)[0]
    d = model.sample_data(jax.random.PRNGKey(5), p)
    np.testing.assert_array_equal(d["x"], X)  # covariates kept
    assert not np.allclose(d["y"], y)


def test_hvp(setup):
    _, _, model = setup
    f = model.log_density_fn("joint", params_as="free")
    n = model.params_pattern.free_length()
    x, v = jnp.ones(n) * 0.1, jnp.arange(n, dtype=float)
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
    free = pg.folded_draws_to_free(pattern=model.params_pattern, draws=draws)
    idata2 = model.to_arviz(free, params_as="free")
    np.testing.assert_allclose(np.asarray(idata2["posterior"]["sigma2"]), np.asarray(draws["sigma2"]), rtol=1e-10)


def _check_distributions_likelihood(model, dists_one):
    other = mg.ExchangeableModel(params_pattern=model.params_pattern, prior=model.prior,
                                 likelihood=mg.exchangeable_dgp_from_distributions(dists_one),
                                 data=model.data, hyper=model.hyper)
    assert_models_match(model, other)
    p = model.params_pattern.random(jax.random.PRNGKey(7))
    d = other.sample_data(jax.random.PRNGKey(8), p)
    assert d["y"].shape == model.data["y"].shape and not np.allclose(d["y"], model.data["y"])
    check_model_contract(other)


def test_distributions_likelihood_numpyro(setup):
    """Writing the likelihood once as distribution objects gives both its
    log density and its sampler."""
    if npdist is None:
        pytest.skip("numpyro unavailable")
    _check_distributions_likelihood(
        setup[2], lambda params, datum, hyper: {"y": npdist.Normal(datum["x"] @ params["beta"],
                                                                   jnp.sqrt(params["sigma2"]))})


def test_distributions_likelihood_tfp(setup):
    if tfp is None:
        pytest.skip("tensorflow_probability (JAX substrate) unavailable")
    tfd = tfp.distributions
    _check_distributions_likelihood(
        setup[2], lambda params, datum, hyper: {"y": tfd.Normal(datum["x"] @ params["beta"],
                                                                jnp.sqrt(params["sigma2"]))})


def test_jacobian_defaults_and_overrides(setup):
    _, _, model = setup
    # Defaults: the Jacobian goes with densities over the parameters (joint, prior),
    # not with the likelihood; `jacobian=` turns it on or off for any of them.
    u = jax.random.normal(jax.random.PRNGKey(3), (model.params_pattern.free_length(),))
    ldj = model.params_pattern.free_to_folded_log_abs_det(u)
    assert jnp.abs(ldj) > 1e-3
    for which, default_on in [("joint", True), ("prior", True), ("lik", False)]:
        with_j = model.log_density_fn(which, jacobian=True)(u)
        without_j = model.log_density_fn(which, jacobian=False)(u)
        default = model.log_density_fn(which)(u)
        assert jnp.allclose(with_j - without_j, ldj, rtol=1e-12, atol=1e-12)
        assert jnp.allclose(default, with_j if default_on else without_j, rtol=1e-12, atol=1e-12)
