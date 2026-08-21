import jax.numpy as jnp
import numpy as np
import pytest

import posteriordb_jax  # noqa: F401
from posteriordb_jax.constraints import ConstraintType, ParameterSpec


def test_unconstrained_scalar():
    s = ParameterSpec("mu", (), ConstraintType.UNCONSTRAINED)
    assert s.constrained_size == 1
    assert s.free_size == 1
    x = jnp.array([0.7])
    y = s.bijector.forward(x)
    np.testing.assert_allclose(y, x)


def test_lower_bound_vector():
    s = ParameterSpec("sigma", (3,), ConstraintType.LOWER, bounds=(0.0, None))
    assert s.free_size == 3
    x = jnp.array([-1.0, 0.0, 2.0])
    y = s.bijector.forward(x)
    assert np.all(np.asarray(y) > 0)
    x_back = s.bijector.inverse(y)
    np.testing.assert_allclose(x_back, x, atol=1e-6)


def test_simplex_spec():
    s = ParameterSpec("theta", (4,), ConstraintType.SIMPLEX)
    assert s.constrained_size == 4
    assert s.free_size == 3
    assert s.matches_stan_unconstrained
    assert s.free_shape == (3,)


def test_simplex_array_spec():
    """Stan's `array[3] simplex[4] theta` -- shape (M, K) with M=3 "array
    of" instances, each a 4-simplex."""
    s = ParameterSpec("theta", (3, 4), ConstraintType.SIMPLEX)
    assert s.constrained_size == 12
    assert s.free_size == 9  # M * (K - 1) = 3 * 3
    assert s.free_shape == (3, 3)
    assert s.flatten_order == "F"
    assert s.matches_stan_unconstrained

    x = jnp.zeros((9,)).reshape(3, 3)
    y = s.bijector.forward(x)
    assert y.shape == (3, 4)
    np.testing.assert_allclose(np.sum(np.asarray(y), axis=-1), np.ones(3), atol=1e-10)


def test_cholesky_factor_corr_spec():
    s = ParameterSpec("L", (3, 3), ConstraintType.CHOLESKY_FACTOR_CORR)
    assert s.constrained_size == 9
    assert s.free_size == 3
    assert s.flatten_order == "F"
    x = jnp.array([0.1, -0.2, 0.3])
    y = s.bijector.forward(x)
    assert y.shape == (3, 3)


def test_bad_shape_raises():
    # (2, 3) is a *valid* SIMPLEX shape now (an array[2] simplex[3]) -- only
    # a 0-D shape is invalid for SIMPLEX (there's always at least a K axis).
    with pytest.raises(ValueError):
        ParameterSpec("theta", (), ConstraintType.SIMPLEX)
    with pytest.raises(ValueError):
        ParameterSpec("theta", (2, 3), ConstraintType.ORDERED)
    with pytest.raises(ValueError):
        ParameterSpec("L", (2, 3), ConstraintType.CHOLESKY_FACTOR_CORR)


def test_deferred_constraint_raises_on_bijector_access():
    s = ParameterSpec("u", (3,), ConstraintType.UNIT_VECTOR)
    assert not s.matches_stan_unconstrained
    with pytest.raises(NotImplementedError):
        _ = s.bijector
