import jax
import jax.numpy as jnp
import numpy as np
import pytest

import posteriordb_jax  # noqa: F401
from posteriordb_jax.constraints import ConstraintType, ParameterSpec
from posteriordb_jax.flatten import ModelSpec

jax.config.update("jax_enable_x64", True)


def _dogs_like_spec():
    return ModelSpec(
        (
            ParameterSpec("mu", (2,), ConstraintType.UNCONSTRAINED),
            ParameterSpec("sigma", (2,), ConstraintType.LOWER, bounds=(0.0, None)),
            ParameterSpec("L", (2, 2), ConstraintType.CHOLESKY_FACTOR_CORR),
            ParameterSpec("z", (5, 2), ConstraintType.UNCONSTRAINED),
        )
    )


def test_free_and_constrained_dims():
    ms = _dogs_like_spec()
    assert ms.free_dim == 2 + 2 + 1 + 10
    assert ms.constrained_dim == 2 + 2 + 4 + 10


def test_constrain_unconstrain_round_trip(rng=np.random.default_rng(0)):
    ms = _dogs_like_spec()
    x = jnp.asarray(rng.normal(size=ms.free_dim))
    params, logdet_fwd = ms.constrain(x)
    assert set(params) == {"mu", "sigma", "L", "z"}
    assert params["mu"].shape == (2,)
    assert params["L"].shape == (2, 2)
    assert params["z"].shape == (5, 2)

    x_back, logdet_inv = ms.unconstrain(params)
    np.testing.assert_allclose(x_back, x, atol=1e-6)
    np.testing.assert_allclose(logdet_fwd, -logdet_inv, atol=1e-6)


def test_constrain_batched(rng=np.random.default_rng(1)):
    ms = _dogs_like_spec()
    x = jnp.asarray(rng.normal(size=(7, ms.free_dim)))
    params, logdet = ms.constrain(x)
    assert params["z"].shape == (7, 5, 2)
    assert params["L"].shape == (7, 2, 2)
    assert logdet.shape == (7,)
    # matches per-draw loop
    params0, logdet0 = ms.constrain(x[0])
    np.testing.assert_allclose(params["z"][0], params0["z"], atol=1e-8)
    np.testing.assert_allclose(logdet[0], logdet0, atol=1e-8)


def test_flatten_constrained_round_trip(rng=np.random.default_rng(2)):
    ms = _dogs_like_spec()
    x = jnp.asarray(rng.normal(size=ms.free_dim))
    params, _ = ms.constrain(x)
    flat = ms.flatten_constrained(params)
    assert flat.shape == (ms.constrained_dim,)
    params_back = ms.unflatten_constrained(flat)
    for name in params:
        np.testing.assert_allclose(params_back[name], params[name], atol=1e-8)


def test_matrix_column_major_element_keys():
    ms = ModelSpec((ParameterSpec("z", (3, 2), ConstraintType.UNCONSTRAINED),))
    z = jnp.arange(6, dtype=jnp.float64).reshape(3, 2)  # row-major fill for the test data itself
    pdb_dict = ms.to_posteriordb_dict({"z": z})
    # column-major (Fortran) order: z[1,1], z[2,1], z[3,1], z[1,2], z[2,2], z[3,2]
    expected_keys = ["z[1,1]", "z[2,1]", "z[3,1]", "z[1,2]", "z[2,2]", "z[3,2]"]
    assert list(pdb_dict.keys()) == expected_keys
    np.testing.assert_allclose(
        [pdb_dict[k] for k in expected_keys],
        [z[0, 0], z[1, 0], z[2, 0], z[0, 1], z[1, 1], z[2, 1]],
    )

    z_back = ms.from_posteriordb_dict(pdb_dict)["z"]
    np.testing.assert_allclose(z_back, z)


def _simplex_array_spec():
    return ModelSpec(
        (
            ParameterSpec("mu", (), ConstraintType.UNCONSTRAINED),
            ParameterSpec("theta", (3, 4), ConstraintType.SIMPLEX),  # array[3] simplex[4]
        )
    )


def test_simplex_array_constrain_unconstrain_round_trip(rng=np.random.default_rng(3)):
    ms = _simplex_array_spec()
    assert ms.free_dim == 1 + 3 * 3
    assert ms.constrained_dim == 1 + 3 * 4

    x = jnp.asarray(rng.normal(size=ms.free_dim))
    params, logdet_fwd = ms.constrain(x)
    assert params["theta"].shape == (3, 4)
    np.testing.assert_allclose(np.sum(np.asarray(params["theta"]), axis=-1), np.ones(3), atol=1e-10)

    x_back, logdet_inv = ms.unconstrain(params)
    np.testing.assert_allclose(x_back, x, atol=1e-6)
    np.testing.assert_allclose(logdet_fwd, -logdet_inv, atol=1e-6)


def test_simplex_array_constrain_batched(rng=np.random.default_rng(4)):
    ms = _simplex_array_spec()
    x = jnp.asarray(rng.normal(size=(6, ms.free_dim)))
    params, logdet = ms.constrain(x)
    assert params["theta"].shape == (6, 3, 4)
    assert logdet.shape == (6,)
    params0, logdet0 = ms.constrain(x[0])
    np.testing.assert_allclose(params["theta"][0], params0["theta"], atol=1e-8)
    np.testing.assert_allclose(logdet[0], logdet0, atol=1e-8)


def test_vector_element_keys():
    ms = ModelSpec((ParameterSpec("theta", (3,), ConstraintType.UNCONSTRAINED),))
    theta = jnp.array([1.0, 2.0, 3.0])
    pdb_dict = ms.to_posteriordb_dict({"theta": theta})
    assert list(pdb_dict.keys()) == ["theta[1]", "theta[2]", "theta[3]"]
    theta_back = ms.from_posteriordb_dict(pdb_dict)["theta"]
    np.testing.assert_allclose(theta_back, theta)
