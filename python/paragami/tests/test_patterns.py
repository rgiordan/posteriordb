import jax
import jax.numpy as jnp
import numpy as np
import pytest

import paragami as pg

jax.config.update("jax_enable_x64", True)


def _patterns():
    return {
        "scalar": pg.NumericScalarPattern(),
        "lb": pg.NumericArrayPattern((2, 3), lb=-1.0),
        "ub": pg.NumericVectorPattern(4, ub=2.0),
        "lub": pg.NumericArrayPattern((3,), lb=0.0, ub=5.0),
        "lub_F": pg.NumericArrayPattern((2, 3), lb=0.0, ub=1.0, order="F"),
        "lb_tfp": pg.NumericVectorPattern(3, lb=1.0, bijector="tfp"),
        "ub_tfp": pg.NumericScalarPattern(ub=2.0, bijector="tfp"),
        "affine": pg.NumericVectorPattern(3, offset=1.0, multiplier=2.0),
        "affine_array": pg.NumericArrayPattern((2, 2), offset=[[1.0, 2.0], [3.0, 4.0]],
                                               multiplier=[[0.5, 1.0], [2.0, 3.0]], order="F"),
        "ordered": pg.OrderedVectorPattern(4),
        "pos_ordered": pg.OrderedVectorPattern(3, positive=True),
        "unit_vector": pg.UnitVectorPattern(3),
        "simplex": pg.SimplexArrayPattern(4),
        "simplex_tfp": pg.SimplexArrayPattern(4, bijector="tfp"),
        "simplex_array": pg.SimplexArrayPattern(3, array_shape=(2, 2)),
        "column_stochastic": pg.StochasticMatrixPattern(3, 2, kind="column"),
        "row_stochastic_tfp": pg.StochasticMatrixPattern(3, 4, kind="row", bijector="tfp"),
        "sum_to_zero": pg.SumToZeroVectorPattern(4),
        "sum_to_zero_matrix": pg.SumToZeroMatrixPattern(3, 4),
        "chol_cov": pg.CholeskyCovPattern(3),
        "chol_cov_tfp": pg.CholeskyCovPattern(3, bijector="tfp"),
        "psd": pg.PSDSymmetricMatrixPattern(3),
        "psd_tfp": pg.PSDSymmetricMatrixPattern(3, bijector="tfp"),
        "chol_corr": pg.CholeskyCorrPattern(4),
        "chol_corr_tfp": pg.CholeskyCorrPattern(4, bijector="tfp"),
        "corr": pg.CorrMatrixPattern(4),
        "corr_tfp": pg.CorrMatrixPattern(4, bijector="tfp"),
        "array_of_psd": pg.PatternArray((2,), pg.PSDSymmetricMatrixPattern(2)),
        "dict": pg.PatternDict(
            a=pg.NumericScalarPattern(lb=0.0),
            b=pg.SimplexArrayPattern(3),
            c=pg.PatternDict(d=pg.OrderedVectorPattern(2)),
        ),
    }


def _tril(flat, size, k=0):
    rows, cols = np.tril_indices(size, k=k)
    return flat.reshape(size, size)[rows, cols]


# Coordinates of the flat vector that parameterize each pattern, so the
# free -> chart map is square and its log det can be checked by autodiff,
# plus the constant each convention adds (log(K)/2 per simplex in TFP's
# SoftmaxCentered). Patterns whose "log det" is not a Jacobian determinant
# (unit vector, sum to zero, TFP's correlation matrix chain) are checked
# against Stan and TFP in test_against_stan_and_tfp.py instead.
_CHARTS = {
    "simplex": (lambda p, f: f[:-1], 0.0),
    "simplex_tfp": (lambda p, f: f[:-1], 0.5 * np.log(4.0)),
    "simplex_array": (lambda p, f: f.reshape(-1, 3)[:, :-1].ravel(), 0.0),
    "column_stochastic": (lambda p, f: f.reshape(3, 2)[:-1, :].ravel(order="F"), 0.0),
    "row_stochastic_tfp": (lambda p, f: f.reshape(3, 4)[:, :-1].ravel(order="F"), 3 * 0.5 * np.log(4.0)),
    "chol_cov": (lambda p, f: _tril(f, 3), 0.0),
    "chol_cov_tfp": (lambda p, f: _tril(f, 3), 0.0),
    "psd": (lambda p, f: _tril(f, 3), 0.0),
    "psd_tfp": (lambda p, f: _tril(f, 3), 0.0),
    "chol_corr": (lambda p, f: _tril(f, 4, k=-1), 0.0),
    "chol_corr_tfp": (lambda p, f: _tril(f, 4, k=-1), 0.0),
    "corr": (lambda p, f: _tril(f, 4, k=-1), 0.0),
    "array_of_psd": (lambda p, f: f.reshape(2, 2, 2)[:, [0, 1, 1], [0, 0, 1]].ravel(), 0.0),
    "dict": (lambda p, f: jnp.concatenate([f[:1], f[1:3], f[4:6]]), 0.0),  # a, first 2 of simplex b, d
}
_NOT_JACOBIANS = {"unit_vector", "sum_to_zero", "sum_to_zero_matrix", "corr_tfp"}


@pytest.mark.parametrize("name", list(_patterns()))
def test_round_trips(name):
    pattern = _patterns()[name]
    u = jax.random.normal(jax.random.PRNGKey(0), (pattern.get_free_length(),))
    folded = pattern.free_to_folded(u)
    ok, msg = pattern.validate_folded(folded)
    assert ok, msg
    expected = u / jnp.linalg.norm(u) if name == "unit_vector" else u  # Stan's unit_vector_free returns x
    np.testing.assert_allclose(pattern.folded_to_free(folded), expected, atol=1e-10)
    flat = pattern.folded_to_flat(folded)
    assert flat.shape == (pattern.get_flat_length(),)
    refolded = pattern.flat_to_folded(flat)
    jax.tree_util.tree_map(lambda a, b: np.testing.assert_allclose(a, b, atol=1e-12), folded, refolded)
    assert len(pattern.get_free_names()) == pattern.get_free_length() or pattern.get_free_length() == 1
    assert len(pattern.get_flat_names()) == pattern.get_flat_length() or pattern.get_flat_length() == 1


@pytest.mark.parametrize("name", [n for n in _patterns() if n not in _NOT_JACOBIANS])
def test_log_det_matches_autodiff(name):
    pattern = _patterns()[name]
    u = 0.5 * jax.random.normal(jax.random.PRNGKey(1), (pattern.get_free_length(),))
    chart, constant = _CHARTS.get(name, (lambda p, f: f, 0.0))

    def to_chart(u_):
        return chart(pattern, pattern.folded_to_flat(pattern.free_to_folded(u_)))

    jac = jax.jacfwd(to_chart)(u)
    assert jac.shape == (u.shape[0], u.shape[0])
    expected = jnp.linalg.slogdet(jac)[1] + constant
    np.testing.assert_allclose(pattern.free_to_folded_log_abs_det(u), expected, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("name", list(_patterns()))
def test_direct_conversions_match_fold_and_flatten(name):
    pattern = _patterns()[name]
    u = jax.random.normal(jax.random.PRNGKey(4), (pattern.get_free_length(),))
    folded = pattern.free_to_folded(u)
    same = lambda a, b: jax.tree_util.tree_map(lambda x, y: np.testing.assert_allclose(x, y, atol=1e-12), a, b)
    same(folded, pattern.fold(u, from_free=True))
    folded_again, log_det = pattern.free_to_folded_with_log_det(u)
    same(folded, folded_again)
    np.testing.assert_allclose(log_det, pattern.free_to_folded_log_abs_det(u))
    np.testing.assert_allclose(pattern.folded_to_free(folded), pattern.flatten(folded, to_free=True))
    flat = pattern.folded_to_flat(folded)
    np.testing.assert_allclose(flat, pattern.flatten(folded, to_free=False))
    same(pattern.flat_to_folded(flat), pattern.fold(flat, from_free=False))


@pytest.mark.parametrize("name", list(_patterns()))
def test_generate_code_round_trip(name):
    pattern = _patterns()[name]
    assert eval(pattern.generate_code(), {"pg": pg}) == pattern
    assert eval(pattern.generate_code(module=""), vars(pg)) == pattern


def test_transforms_are_jittable_and_vmappable():
    pattern = _patterns()["dict"]
    us = jax.random.normal(jax.random.PRNGKey(2), (5, pattern.get_free_length()))
    folded, log_det = jax.jit(jax.vmap(pattern.free_to_folded_with_log_det))(us)
    assert folded["b"].shape == (5, 3)
    assert log_det.shape == (5,)
    g = jax.grad(pattern.free_to_folded_log_abs_det)(us[0])
    assert g.shape == (pattern.get_free_length(),)


def test_validation_errors():
    p = pg.PatternDict(sigma=pg.NumericScalarPattern(lb=0.0), w=pg.SimplexArrayPattern(3))
    ok, msg = p.validate_folded({"sigma": -1.0, "w": jnp.array([0.2, 0.3, 0.5])})
    assert not ok and "lb" in msg
    ok, msg = p.validate_folded({"sigma": 1.0, "w": jnp.array([0.2, 0.3, 0.6])})
    assert not ok and "sum to one" in msg
    ok, msg = p.validate_folded({"sigma": 1.0})
    assert not ok and "missing" in msg
    with pytest.raises(pg.ValidationError):
        p.free_to_folded(jnp.zeros(4))  # wrong length (free length is 3)
    with pytest.raises(pg.ValidationError):
        p.folded_to_free({"sigma": -1.0, "w": jnp.ones(3) / 3})
    # Value checks are skipped under tracing, shape checks are not.
    jax.jit(lambda s: p.folded_to_free({"sigma": s, "w": jnp.ones(3) / 3}))(-1.0)


def test_validation_switch():
    inner = pg.PatternDict(w=pg.SimplexArrayPattern(3))
    p = pg.PatternDict(sigma=pg.NumericScalarPattern(lb=0.0), inner=inner)
    bad = {"sigma": -1.0, "inner": {"w": jnp.ones(3) / 3}}
    assert p.disable_validation() is p
    assert not p.validate and not p["sigma"].validate and not inner.validate and not inner["w"].validate
    p.folded_to_free(bad)  # no check
    p.free_to_folded(jnp.zeros(3))  # no length check either
    assert not p.validate_folded(bad)[0]  # an explicit check still runs
    p["tau"] = pg.NumericScalarPattern(lb=0.0)  # new children follow the container
    assert not p["tau"].validate
    p.enable_validation()
    assert p["tau"].validate and inner["w"].validate
    with pytest.raises(pg.ValidationError):
        p.folded_to_free({**bad, "tau": 1.0})
    arr = pg.PatternArray((2,), pg.SimplexArrayPattern(3)).disable_validation()
    assert not arr.base_pattern.validate


def test_free_or_flat_must_be_given():
    p = pg.NumericScalarPattern(lb=0.0)
    with pytest.raises(TypeError):
        p.fold(jnp.zeros(1))
    with pytest.raises(TypeError):
        p.flatten(1.0)


def test_affine_and_bijector_arguments():
    with pytest.raises(ValueError):
        pg.NumericScalarPattern(lb=0.0, multiplier=2.0)
    with pytest.raises(ValueError):
        pg.NumericVectorPattern(2, multiplier=[1.0, -1.0])
    with pytest.raises(ValueError):
        pg.NumericArrayPattern((2, 2), offset=[1.0, 2.0])
    with pytest.raises(ValueError):
        pg.SimplexArrayPattern(3, bijector="pymc")
    p = pg.NumericVectorPattern(2, offset=[1.0, 2.0], multiplier=[2.0, 4.0])
    np.testing.assert_allclose(p.free_to_folded(jnp.ones(2)), [3.0, 6.0])
    np.testing.assert_allclose(p.free_to_folded_log_abs_det(jnp.ones(2)), np.log(8.0))
    assert p != pg.NumericVectorPattern(2, offset=[1.0, 2.0])
    assert pg.NumericVectorPattern(2, lb=0.0) != pg.NumericVectorPattern(2, lb=0.0, bijector="tfp")


def test_column_major_order():
    p = pg.NumericArrayPattern((2, 3), order="F")
    x = jnp.arange(6.0).reshape(2, 3)
    np.testing.assert_array_equal(p.folded_to_flat(x), [0, 3, 1, 4, 2, 5])
    assert p.get_flat_names()[:2] == ["[0,0]", "[1,0]"]


def test_function_wrappers():
    p = pg.PatternDict(mu=pg.NumericScalarPattern(), sigma=pg.NumericScalarPattern(lb=0.0))
    y = jnp.array([1.0, 2.0])
    folded = {"mu": 1.0, "sigma": 1.0}
    u = jnp.array([1.0, 0.0])  # free: sigma = exp(0)

    def f(params, y):
        return jnp.sum((y - params["mu"]) ** 2) / params["sigma"]

    g = pg.function_input_folded_to_free(f, patterns=p, argnums=0)
    assert g(u, y) == f(folded, y)
    assert jax.grad(g)(u, y).shape == (2,)
    assert pg.function_input_free_to_folded(g, patterns=p)(folded, y) == g(u, y)
    g_flat = pg.function_input_folded_to_flat(f, patterns=p)
    assert g_flat(jnp.array([1.0, 1.0]), y) == f(folded, y)
    assert pg.function_input_flat_to_folded(g_flat, patterns=p)(folded, y) == f(folded, y)
    # Two arguments with their own patterns.
    q = pg.NumericVectorPattern(2, lb=0.0)
    h = pg.function_input_folded_to_free(f, patterns=[p, q], argnums=[0, 1])
    np.testing.assert_allclose(h(u, jnp.log(y)), f(folded, y))

    make = lambda x: {"mu": x, "sigma": jnp.exp(x)}
    np.testing.assert_allclose(pg.function_output_folded_to_free(make, pattern=p)(0.5), [0.5, 0.5])
    np.testing.assert_allclose(pg.function_output_folded_to_flat(make, pattern=p)(0.0), [0.0, 1.0])
    to_free = pg.function_output_folded_to_free(make, pattern=p)
    np.testing.assert_allclose(pg.function_output_free_to_folded(to_free, pattern=p)(0.5)["sigma"], jnp.exp(0.5))
    to_flat = pg.function_output_folded_to_flat(make, pattern=p)
    np.testing.assert_allclose(pg.function_output_flat_to_folded(to_flat, pattern=p)(0.5)["sigma"], jnp.exp(0.5))


def test_empty_random_and_equality():
    p = _patterns()["dict"]
    assert p == _patterns()["dict"]
    assert p != pg.PatternDict(a=pg.NumericScalarPattern())
    r = p.draw_random(jax.random.PRNGKey(0))
    assert p.validate_folded(r)[0]
    assert np.all(np.isnan(p.make_empty(valid=False)["a"]))


def test_nested_dict_names():
    p = pg.PatternDict(a=pg.PatternDict(b=pg.NumericVectorPattern(2), c=pg.NumericScalarPattern()),
                       d=pg.NumericScalarPattern())
    assert p.get_flat_names() == ["a.b[0]", "a.b[1]", "a.c", "d"]


def test_unconstrained_pattern_like():
    value = {
        "beta": jnp.array([1.0, 2.0, 3.0]),
        "sigma": 0.5,
        "nested": {"cov": np.eye(2), "count": 3},
        "w": [jnp.zeros(2), 1.0],
    }
    p = pg.make_unconstrained_pattern_like(value)
    assert list(p.keys()) == ["beta", "sigma", "nested", "w"]
    assert list(p["w"].keys()) == ["w_0", "w_1"]
    assert p.get_free_length() == p.get_flat_length() == 3 + 1 + 4 + 1 + 2 + 1
    x = p.folded_to_free({**value, "w": {"w_0": value["w"][0], "w_1": value["w"][1]}})
    folded = p.free_to_folded(x)
    np.testing.assert_allclose(folded["nested"]["cov"], np.eye(2))
    assert folded["sigma"].shape == () and folded["w"]["w_0"].shape == (2,)
    assert "nested.cov[1,1]" in p.get_flat_names()

    # Names for unnamed containers come from default_name; a bare leaf is a single pattern.
    assert list(pg.make_unconstrained_pattern_like((1.0, jnp.ones(3)), default_name="theta").keys()) == ["theta_0", "theta_1"]
    leaf = pg.make_unconstrained_pattern_like(jnp.ones((2, 3)))
    assert type(leaf) is pg.NumericArrayPattern and leaf.get_free_length() == 6
    assert type(pg.make_unconstrained_pattern_like(1.0)) is pg.NumericScalarPattern
    with pytest.raises(TypeError):
        pg.make_unconstrained_pattern_like({"a": "text"})


def test_unconstrained_pattern_code():
    value = {"beta": jnp.zeros(3), "s": 1.0, "not an identifier": jnp.zeros((2, 2))}
    code = pg.make_unconstrained_pattern_like(value).generate_code()
    assert "beta=" not in code and "'not an identifier':" in code
    assert eval(code, {"pg": pg}) == pg.make_unconstrained_pattern_like(value)
    code = pg.make_unconstrained_pattern_like({"beta": jnp.zeros(3), "inner": {"s": 1.0}}).generate_code()
    assert code == (
        "pg.PatternDict(\n"
        "    beta=pg.NumericVectorPattern(3),\n"
        "    inner=pg.PatternDict(\n"
        "        s=pg.NumericScalarPattern(),\n"
        "    ),\n"
        ")"
    )


def test_arviz_round_trip():
    pytest.importorskip("arviz")
    p = pg.PatternDict(beta=pg.NumericVectorPattern(2), sigma=pg.NumericScalarPattern(lb=0.0),
                       inner=pg.PatternDict(w=pg.SimplexArrayPattern(3), cov=pg.PSDSymmetricMatrixPattern(2)))
    free = jax.random.normal(jax.random.PRNGKey(0), (2, 5, p.get_free_length()))
    idata = pg.free_draws_to_arviz(pattern=p, draws=free, coords={"coef": ["a", "b"]}, dims={"beta": ["coef"]},
                                   extra_groups={"observed_data": {"y": np.zeros(4)}})
    post = idata["posterior"] if not hasattr(idata, "posterior") else idata.posterior
    assert set(post.data_vars) == {"beta", "sigma", "inner.w", "inner.cov"}
    assert post["inner.cov"].shape == (2, 5, 2, 2) and list(post["beta"].coords["coef"].values) == ["a", "b"]
    np.testing.assert_allclose(pg.arviz_to_free_draws(pattern=p, data=idata), free, atol=1e-10)
    folded = pg.arviz_to_folded_draws(pattern=p, data=idata)
    np.testing.assert_allclose(folded["inner"]["w"].sum(-1), 1.0)
    flat = pg.folded_draws_to_flat(pattern=p, draws=folded)
    assert flat.shape == (2, 5, p.get_flat_length())
    again = pg.folded_draws_to_arviz(pattern=p, draws=pg.flat_draws_to_folded(pattern=p, draws=flat), group="prior")
    np.testing.assert_allclose(pg.arviz_to_free_draws(pattern=p, data=again, group="prior"), free, atol=1e-10)
    # A single (non-dict) pattern is one variable named `name`.
    q = pg.SimplexArrayPattern(3)
    d = pg.free_draws_to_arviz(pattern=q, draws=free[..., :2], name="w")
    np.testing.assert_allclose(pg.arviz_to_free_draws(pattern=q, data=d, name="w"), free[..., :2], atol=1e-10)
