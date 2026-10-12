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
        "ordered": pg.OrderedVectorPattern(4),
        "pos_ordered": pg.OrderedVectorPattern(3, positive=True),
        "simplex": pg.SimplexArrayPattern(4),
        "simplex_array": pg.SimplexArrayPattern(3, array_shape=(2, 2)),
        "chol_cov": pg.CholeskyCovPattern(3),
        "psd": pg.PSDSymmetricMatrixPattern(3),
        "chol_corr": pg.CholeskyCorrPattern(4),
        "array_of_psd": pg.PatternArray((2,), pg.PSDSymmetricMatrixPattern(2)),
        "dict": pg.PatternDict(
            a=pg.NumericScalarPattern(lb=0.0),
            b=pg.SimplexArrayPattern(3),
            c=pg.PatternDict(d=pg.OrderedVectorPattern(2)),
        ),
    }


# Coordinates of the flat vector that parameterize each pattern's manifold,
# so the free -> chart map is square and its log det can be checked.
def _chart(name, pattern, flat):
    if name.startswith("simplex"):
        k = pattern.simplex_size
        return flat.reshape(-1, k)[:, :-1].ravel()
    if name in ("chol_cov", "psd"):
        r, c = np.tril_indices(pattern.size)
        return flat.reshape(pattern.shape)[r, c]
    if name == "chol_corr":
        r, c = np.tril_indices(pattern.size, -1)
        return flat.reshape(pattern.shape)[r, c]
    if name == "array_of_psd":
        r, c = np.tril_indices(2)
        return flat.reshape(2, 2, 2)[:, r, c].ravel()
    if name == "dict":
        # a (1), b simplex (first 2 of 3), d (2)
        return jnp.concatenate([flat[:1], flat[1:3], flat[4:6]])
    return flat


@pytest.mark.parametrize("name", list(_patterns()))
def test_round_trips(name):
    pattern = _patterns()[name]
    u = jax.random.normal(jax.random.PRNGKey(0), (pattern.free_length(),))
    folded = pattern.fold(u, from_free=True)
    ok, msg = pattern.validate_folded(folded)
    assert ok, msg
    np.testing.assert_allclose(pattern.flatten(folded, to_free=True), u, atol=1e-10)
    flat = pattern.flatten(folded, to_free=False)
    assert flat.shape == (pattern.flat_length(),)
    refolded = pattern.fold(flat, from_free=False)
    jax.tree_util.tree_map(lambda a, b: np.testing.assert_allclose(a, b, atol=1e-12), folded, refolded)
    assert len(pattern.free_names()) == pattern.free_length() or pattern.free_length() == 1
    assert len(pattern.flat_names()) == pattern.flat_length() or pattern.flat_length() == 1


@pytest.mark.parametrize("name", list(_patterns()))
def test_log_det_matches_autodiff(name):
    pattern = _patterns()[name]
    u = 0.5 * jax.random.normal(jax.random.PRNGKey(1), (pattern.free_length(),))

    def chart(u_):
        return _chart(name, pattern, pattern.flatten(pattern.fold(u_, from_free=True), to_free=False, validate_value=False))

    jac = jax.jacfwd(chart)(u)
    expected = jnp.linalg.slogdet(jac)[1]
    _, logdet = pattern.free_to_folded_with_log_det(u)
    np.testing.assert_allclose(logdet, expected, rtol=1e-8, atol=1e-8)


def test_transforms_are_jittable_and_vmappable():
    pattern = _patterns()["dict"]
    us = jax.random.normal(jax.random.PRNGKey(2), (5, pattern.free_length()))
    folded = jax.jit(jax.vmap(lambda u: pattern.free_to_folded_with_log_det(u)))(us)
    assert folded[0]["b"].shape == (5, 3)
    assert folded[1].shape == (5,)
    g = jax.grad(lambda u: pattern.free_to_folded_with_log_det(u)[1])(us[0])
    assert np.all(np.isfinite(g))


def test_validation_errors():
    p = pg.PatternDict(sigma=pg.NumericScalarPattern(lb=0.0), w=pg.SimplexArrayPattern(3))
    ok, msg = p.validate_folded({"sigma": -1.0, "w": jnp.array([0.2, 0.3, 0.5])})
    assert not ok and "sigma" in msg
    ok, msg = p.validate_folded({"sigma": 1.0, "w": jnp.array([0.2, 0.3, 0.6])})
    assert not ok and "sum to one" in msg
    ok, msg = p.validate_folded({"sigma": 1.0})
    assert not ok and "missing" in msg
    with pytest.raises(pg.ValidationError):
        p.fold(jnp.zeros(4), from_free=True)  # wrong length (free length is 3)
    # Value checks are skipped under tracing, shape checks are not.
    jax.jit(lambda s: p.flatten({"sigma": s, "w": jnp.ones(3) / 3}, to_free=True))(-1.0)


def test_column_major_order():
    p = pg.NumericArrayPattern((2, 3), order="F")
    x = jnp.arange(6.0).reshape(2, 3)
    np.testing.assert_array_equal(p.flatten(x, to_free=False), [0, 3, 1, 4, 2, 5])
    assert p.flat_names()[:2] == ["[0,0]", "[1,0]"]


def test_function_wrappers():
    p = pg.PatternDict(mu=pg.NumericScalarPattern(), sigma=pg.NumericScalarPattern(lb=0.0))
    y = jnp.array([1.0, 2.0])
    folded = {"mu": 1.0, "sigma": 1.0}
    u = jnp.array([1.0, 0.0])  # free: sigma = exp(0)

    def f(params, y):
        return jnp.sum((y - params["mu"]) ** 2) / params["sigma"]

    g = pg.function_input_folded_to_free(f, p, argnums=0)
    assert g(u, y) == f(folded, y)
    assert jax.grad(g)(u, y).shape == (2,)
    assert pg.function_input_free_to_folded(g, p)(folded, y) == g(u, y)
    g_flat = pg.function_input_folded_to_flat(f, p)
    assert g_flat(jnp.array([1.0, 1.0]), y) == f(folded, y)
    assert pg.function_input_flat_to_folded(g_flat, p)(folded, y) == f(folded, y)
    # Two arguments with their own patterns.
    q = pg.NumericVectorPattern(2, lb=0.0)
    h = pg.function_input_folded_to_free(f, [p, q], argnums=[0, 1])
    np.testing.assert_allclose(h(u, jnp.log(y)), f(folded, y))

    make = lambda x: {"mu": x, "sigma": jnp.exp(x)}
    np.testing.assert_allclose(pg.function_output_folded_to_free(make, p)(0.5), [0.5, 0.5])
    np.testing.assert_allclose(pg.function_output_folded_to_flat(make, p)(0.0), [0.0, 1.0])
    back = pg.function_output_free_to_folded(pg.function_output_folded_to_free(make, p), p)(0.5)
    np.testing.assert_allclose(back["sigma"], jnp.exp(0.5))
    back = pg.function_output_flat_to_folded(pg.function_output_folded_to_flat(make, p), p)(0.5)
    np.testing.assert_allclose(back["sigma"], jnp.exp(0.5))


def test_empty_random_and_equality():
    p = _patterns()["dict"]
    assert p == _patterns()["dict"]
    assert p != pg.PatternDict(a=pg.NumericScalarPattern())
    r = p.random(jax.random.PRNGKey(0))
    assert p.validate_folded(r)[0]
    assert np.all(np.isnan(p.empty(valid=False)["a"]))


@pytest.mark.parametrize("name", list(_patterns()))
def test_direct_conversions_match_fold_and_flatten(name):
    pattern = _patterns()[name]
    u = jax.random.normal(jax.random.PRNGKey(4), (pattern.free_length(),))
    folded = pattern.free_to_folded(u)
    same = lambda a, b: jax.tree_util.tree_map(lambda x, y: np.testing.assert_allclose(x, y, atol=1e-12), a, b)
    same(folded, pattern.fold(u, from_free=True))
    same(folded, pattern.free_to_folded_with_log_det(u)[0])
    np.testing.assert_allclose(pattern.free_to_folded_log_abs_det(u), pattern.free_to_folded_with_log_det(u)[1])
    np.testing.assert_allclose(pattern.folded_to_free(folded), pattern.flatten(folded, to_free=True))
    flat = pattern.folded_to_flat(folded)
    np.testing.assert_allclose(flat, pattern.flatten(folded, to_free=False))
    same(pattern.flat_to_folded(flat), folded)


def test_free_default_and_keyword_errors():
    p = pg.NumericScalarPattern(lb=0.0)
    with pytest.raises(ValueError, match="from_free"):
        p.fold(jnp.zeros(1))
    with pytest.raises(ValueError, match="to_free"):
        p.flatten(1.0)
    q = pg.NumericScalarPattern(lb=0.0, free_default=True)
    np.testing.assert_allclose(q.flatten(1.0), 0.0)
    np.testing.assert_allclose(q.fold(jnp.zeros(1)), 1.0)


def test_nested_dict_names():
    p = pg.PatternDict(a=pg.PatternDict(b=pg.NumericVectorPattern(2), c=pg.NumericScalarPattern()), d=pg.NumericScalarPattern())
    assert p.flat_names() == ["a.b[0]", "a.b[1]", "a.c", "d"]
