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
    log_evidence = cr.eval_log_marginal_likelihood(X, y, model.hyper)
    for k in jax.random.split(jax.random.PRNGKey(1), 5):
        params = model.params_pattern.random(k)
        diff = model.eval_log_joint(params) - cr.eval_exact_log_posterior(params, X, y, model.hyper)
        np.testing.assert_allclose(diff, log_evidence, rtol=1e-10)


def test_posterior_mode_by_optimization(setup):
    """Newton's method on the log joint as a function of the free vector,
    without the Jacobian, finds the exact posterior mode of (beta, sigma2)."""
    X, y, model = setup
    pattern = model.params_pattern
    f = jax.jit(lambda u: model.eval_log_joint(pattern.free_to_folded(u)))
    g = jax.jit(jax.grad(f))
    h = jax.jit(jax.hessian(f))
    x = jnp.zeros(pattern.free_length())
    for _ in range(100):  # damped Newton (gradient step where the Hessian isn't negative definite)
        H = h(x)
        step = -jnp.linalg.solve(H, g(x)) if np.all(np.linalg.eigvalsh(H) < 0) else 0.01 * g(x)
        t = 1.0
        while f(x + t * step) < f(x) and t > 1e-8:
            t /= 2
        x = x + t * step
    params = pattern.free_to_folded(x)
    post = cr.compute_exact_posterior(X, y, model.hyper)
    p = X.shape[1]
    np.testing.assert_allclose(params["beta"], post["m0"], rtol=1e-8)
    # joint mode in sigma2 of N(beta; mn, s2 Vn) IG(s2; an, bn) at beta = mn
    np.testing.assert_allclose(params["sigma2"], post["b0"] / (post["a0"] + 1 + p / 2), rtol=1e-8)


def test_free_and_folded_dgp(setup):
    """free_dgp adds the log Jacobian and converts draws; folded_dgp undoes it."""
    _, _, model = setup
    pattern = model.params_pattern
    free_prior = mg.free_dgp(model.prior, pattern)
    u = jax.random.normal(jax.random.PRNGKey(2), (pattern.free_length(),))
    params, log_det = pattern.free_to_folded_with_log_det(u)
    np.testing.assert_allclose(free_prior.eval_log_density(u, model.hyper),
                               model.eval_log_prior(params) + log_det, rtol=1e-12)
    draw = free_prior.draw(jax.random.PRNGKey(3), model.hyper)
    assert draw.shape == (pattern.free_length(),)
    back = mg.folded_dgp(free_prior, pattern)
    np.testing.assert_allclose(back.eval_log_density(params, model.hyper), model.eval_log_prior(params), rtol=1e-10)
    np.testing.assert_allclose(back.draw(jax.random.PRNGKey(3), model.hyper)["sigma2"],
                               model.draw_prior(jax.random.PRNGKey(3))["sigma2"], rtol=1e-10)
    # The free joint is the free prior plus the likelihood, which has no Jacobian.
    np.testing.assert_allclose(model.eval_free_log_joint(u), model.eval_log_joint(params) + log_det, rtol=1e-12)


def test_function_and_class_styles_agree(setup):
    X, y, model = setup
    other = mg.ExchangeableModel(params_pattern=cr.make_params_pattern(3), prior=cr.NormalInverseGammaPrior(),
                                 likelihood=cr.NormalRegressionLikelihood(), data={"x": X, "y": y},
                                 hyper=cr.make_default_hyper(3))
    assert_models_match(model, other)
    check_model_contract(other)


def test_general_model_matches_exchangeable(setup):
    """The same likelihood written for the whole data set, in a plain Model."""
    X, y, model = setup

    def eval_log_lik(data, params, hyper):
        return jnp.sum(jax.vmap(cr.eval_log_lik_one, in_axes=(0, None, None))(data, params, hyper))

    other = mg.Model(params_pattern=model.params_pattern, prior=model.prior, likelihood=mg.DGP(eval_log_lik),
                     data=model.data, hyper=model.hyper)
    assert_models_match(model, other)
    with pytest.raises(NotImplementedError):
        other.draw_data(jax.random.PRNGKey(0), model.params_pattern.random(jax.random.PRNGKey(1)))


def test_inferred_params_pattern(setup):
    """Without params_pattern, an unconstrained pattern is inferred from a prior draw."""
    _, _, model = setup
    inferred = mg.ExchangeableModel(prior=model.prior, likelihood=model.likelihood, data=model.data,
                                    hyper=model.hyper)
    assert inferred.params_pattern == pg.unconstrained_pattern_like(model.draw_prior(jax.random.PRNGKey(0)),
                                                                     default_name="params")
    assert inferred.params_pattern.free_length() == 4
    with pytest.raises(ValueError):
        mg.ExchangeableModel(likelihood=model.likelihood, data=model.data)  # flat prior cannot draw


def test_missing_pieces_raise(setup):
    _, _, model = setup
    no_prior = mg.ExchangeableModel(params_pattern=model.params_pattern,
                                    likelihood=mg.ExchangeableDGP(cr.eval_log_lik_one), data=model.data)
    params = model.params_pattern.random(jax.random.PRNGKey(0))
    assert no_prior.eval_log_prior(params) == 0.0  # flat prior
    for draw in (lambda: no_prior.draw_prior(jax.random.PRNGKey(0)),
                 lambda: no_prior.draw_data(jax.random.PRNGKey(0), params),
                 lambda: no_prior.draw_joint(jax.random.PRNGKey(0)),
                 lambda: mg.DGP(draw=cr.draw_prior).eval_log_density(params, model.hyper)):
        with pytest.raises(NotImplementedError):
            draw()
    with pytest.raises(TypeError):
        mg.ExchangeableModel(params_pattern=model.params_pattern, likelihood=mg.DGP(), data=model.data)


def test_weights_subsets_and_bootstrap(setup):
    X, y, model = setup
    params = model.params_pattern.random(jax.random.PRNGKey(2))
    singleobs = model.eval_log_lik_singleobs(params)
    prior = model.eval_log_prior(params)
    w = jnp.zeros(model.n_obs).at[:10].set(1.0)
    np.testing.assert_allclose(model.eval_log_joint(params, weights=w), prior + singleobs[:10].sum())
    np.testing.assert_allclose(model.eval_log_joint_subset(params, jnp.arange(10), scale=1.0),
                               prior + singleobs[:10].sum())
    small = {"x": X[:10], "y": y[:10]}
    np.testing.assert_allclose(model.eval_log_lik(params, data=small), singleobs[:10].sum())
    np.testing.assert_allclose(model.likelihood.select_subset(model.data, jnp.arange(10))["y"], y[:10])
    assert model.draw_data(jax.random.PRNGKey(0), params, data=small)["y"].shape == (10,)
    new_hyper = {**model.hyper, "a0": 5.0}
    np.testing.assert_allclose(model.eval_log_prior(params, hyper=new_hyper), cr.eval_log_prior(params, new_hyper))
    bw = model.draw_np_bootstrap_weights(jax.random.PRNGKey(3))
    assert bw.sum() == model.n_obs
    np.testing.assert_array_equal(bw, mg.draw_np_bootstrap_weights(jax.random.PRNGKey(3), model.n_obs))
    # minibatch estimate is unbiased: average over all equal-size batches
    batches = jnp.arange(model.n_obs).reshape(5, 10)
    est = jnp.mean(jnp.stack([model.eval_log_joint_subset(params, b) for b in batches]))
    np.testing.assert_allclose(est, prior + singleobs.sum(), rtol=1e-12)


def test_derivatives_wrt_data_and_hyper(setup):
    X, y, model = setup
    x = jnp.zeros(model.params_pattern.free_length())
    # Derivatives with respect to data, through a paragami pattern for one datum.
    datum_pattern = pg.PatternDict(x=pg.NumericVectorPattern(3), y=pg.NumericScalarPattern())
    flat = jax.vmap(datum_pattern.folded_to_flat)(model.data)
    g = jax.grad(lambda d: model.eval_free_log_joint(x, data=jax.vmap(datum_pattern.flat_to_folded)(d)))(flat)
    assert g.shape == flat.shape and np.all(np.isfinite(g))
    gh = jax.grad(lambda h: model.eval_free_log_joint(x, hyper=h))(model.hyper)
    assert gh["V0"].shape == (3, 3)


def test_draws(setup):
    X, y, model = setup
    p = model.draw_prior(jax.random.PRNGKey(4))
    assert model.params_pattern.validate_folded(p)[0]
    d = model.draw_data(jax.random.PRNGKey(5), p)
    np.testing.assert_array_equal(d["x"], X)  # covariates kept
    assert not np.allclose(d["y"], y)
    params, data = model.draw_joint(jax.random.PRNGKey(6))
    assert model.params_pattern.validate_folded(params)[0] and data["y"].shape == y.shape
    np.testing.assert_allclose(model.draw_marginal_data(jax.random.PRNGKey(6))["y"], data["y"])
    ys = jax.vmap(lambda k: model.draw_marginal_data(k)["y"])(jax.random.split(jax.random.PRNGKey(7), 200))
    assert ys.shape == (200, model.n_obs)


def test_hvp(setup):
    _, _, model = setup
    f = model.eval_free_log_joint
    n = model.params_pattern.free_length()
    x, v = jnp.ones(n) * 0.1, jnp.arange(n, dtype=float)
    np.testing.assert_allclose(mg.make_hvp(f)(x, v), jax.hessian(f)(x) @ v, rtol=1e-8)


def test_arviz_export_with_log_likelihood(setup):
    """ArviZ export is paragami's; the single-observation log likelihood
    goes in as an extra group for LOO."""
    az = pytest.importorskip("arviz")
    X, y, model = setup
    draws = cr.draw_exact_posterior(jax.random.PRNGKey(6), X, y, model.hyper, n_chains=2, n_draws=200)
    free = pg.folded_draws_to_free(pattern=model.params_pattern, draws=draws)
    singleobs = jax.vmap(jax.vmap(model.eval_log_lik_singleobs))(draws)
    idata = pg.free_draws_to_arviz(pattern=model.params_pattern, draws=free,
                                   extra_groups={"log_likelihood": {"y": np.asarray(singleobs)}})
    loo = az.loo(idata)
    assert np.isfinite(float(getattr(loo, "elpd", getattr(loo, "elpd_loo", np.nan))))
    np.testing.assert_allclose(np.asarray(idata["posterior"]["sigma2"]), np.asarray(draws["sigma2"]), rtol=1e-10)


def _check_distributions_likelihood(model, make_dists_one):
    other = mg.ExchangeableModel(params_pattern=model.params_pattern, prior=model.prior,
                                 likelihood=mg.exchangeable_dgp_from_distributions(make_dists_one),
                                 data=model.data, hyper=model.hyper)
    assert_models_match(model, other)
    p = model.params_pattern.random(jax.random.PRNGKey(7))
    d = other.draw_data(jax.random.PRNGKey(8), p)
    assert d["y"].shape == model.data["y"].shape and not np.allclose(d["y"], model.data["y"])
    check_model_contract(other)


def test_distributions_likelihood_numpyro(setup):
    """Writing the likelihood once as distribution objects gives both its
    log density and its sampler."""
    if npdist is None:
        pytest.skip("numpyro unavailable")
    _check_distributions_likelihood(
        setup[2], lambda datum, params, hyper: {"y": npdist.Normal(datum["x"] @ params["beta"],
                                                                   jnp.sqrt(params["sigma2"]))})


def test_distributions_likelihood_tfp(setup):
    if tfp is None:
        pytest.skip("tensorflow_probability (JAX substrate) unavailable")
    tfd = tfp.distributions
    _check_distributions_likelihood(
        setup[2], lambda datum, params, hyper: {"y": tfd.Normal(datum["x"] @ params["beta"],
                                                                jnp.sqrt(params["sigma2"]))})
