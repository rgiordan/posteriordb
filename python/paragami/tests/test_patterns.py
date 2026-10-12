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
    u = jax.random.normal(jax.random.PRNGKey(0), (pattern.flat_length(True),))
    folded = pattern.fold(u, free=True)
    ok, msg = pattern.validate_folded(folded)
    assert ok, msg
    np.testing.assert_allclose(pattern.flatten(folded, free=True), u, atol=1e-10)
    flat = pattern.flatten(folded, free=False)
    assert flat.shape == (pattern.flat_length(False),)
    refolded = pattern.fold(flat, free=False)
    jax.tree_util.tree_map(lambda a, b: np.testing.assert_allclose(a, b, atol=1e-12), folded, refolded)
    assert len(pattern.flat_names(True)) == pattern.flat_length(True) or pattern.flat_length(True) == 1
    assert len(pattern.flat_names(False)) == pattern.flat_length(False) or pattern.flat_length(False) == 1


@pytest.mark.parametrize("name", list(_patterns()))
def test_log_det_matches_autodiff(name):
    pattern = _patterns()[name]
    u = 0.5 * jax.random.normal(jax.random.PRNGKey(1), (pattern.flat_length(True),))

    def chart(u_):
        return _chart(name, pattern, pattern.flatten(pattern.fold(u_, free=True), free=False, validate_value=False))

    jac = jax.jacfwd(chart)(u)
    expected = jnp.linalg.slogdet(jac)[1]
    _, logdet = pattern.fold_with_log_det(u)
    np.testing.assert_allclose(logdet, expected, rtol=1e-8, atol=1e-8)


def test_transforms_are_jittable_and_vmappable():
    pattern = _patterns()["dict"]
    us = jax.random.normal(jax.random.PRNGKey(2), (5, pattern.flat_length(True)))
    folded = jax.jit(jax.vmap(lambda u: pattern.fold_with_log_det(u)))(us)
    assert folded[0]["b"].shape == (5, 3)
    assert folded[1].shape == (5,)
    g = jax.grad(lambda u: pattern.fold_with_log_det(u)[1])(us[0])
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
        p.fold(jnp.zeros(4), free=True)  # wrong length (free length is 3)
    # Value checks are skipped under tracing, shape checks are not.
    jax.jit(lambda s: p.flatten({"sigma": s, "w": jnp.ones(3) / 3}, free=True))(-1.0)


def test_column_major_order():
    p = pg.NumericArrayPattern((2, 3), order="F")
    x = jnp.arange(6.0).reshape(2, 3)
    np.testing.assert_array_equal(p.flatten(x, free=False), [0, 3, 1, 4, 2, 5])
    assert p.flat_names(False)[:2] == ["[0,0]", "[1,0]"]


def test_function_wrappers():
    p = pg.PatternDict(mu=pg.NumericScalarPattern(), sigma=pg.NumericScalarPattern(lb=0.0))

    def f(params, y):
        return jnp.sum((y - params["mu"]) ** 2) / params["sigma"]

    g = pg.FoldFunctionInput(f, p, free=True, argnums=0)
    u = jnp.array([1.0, 0.0])
    assert g(u, jnp.array([1.0, 2.0])) == f({"mu": 1.0, "sigma": 1.0}, jnp.array([1.0, 2.0]))
    assert jax.grad(g)(u, jnp.array([1.0, 2.0])).shape == (2,)
    h = pg.FlattenFunctionInput(g, p, free=True, argnums=0)
    assert h({"mu": 1.0, "sigma": 1.0}, jnp.array([1.0, 2.0])) == g(u, jnp.array([1.0, 2.0]))


def test_empty_random_and_equality():
    p = _patterns()["dict"]
    assert p == _patterns()["dict"]
    assert p != pg.PatternDict(a=pg.NumericScalarPattern())
    r = p.random(jax.random.PRNGKey(0))
    assert p.validate_folded(r)[0]
    assert np.all(np.isnan(p.empty(valid=False)["a"]))
