"""paragami's pure-JAX transforms reproduce the TFP-based, BridgeStan-verified
`ParameterSpec` transforms exactly (same free vector -> same constrained
value and log Jacobian), so paragami inherits their Stan-exactness."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import paragami as pg
import posteriordb_jax  # noqa: F401  (float64)
from posteriordb_jax.constraints import ConstraintType as C
from posteriordb_jax.constraints import ParameterSpec
from posteriordb_jax.flatten import ModelSpec

CASES = [
    (ParameterSpec("a", (3,), C.UNCONSTRAINED), pg.NumericVectorPattern(3)),
    (ParameterSpec("a", (2, 3), C.UNCONSTRAINED), pg.NumericArrayPattern((2, 3), order="F")),
    (ParameterSpec("a", (4,), C.LOWER, bounds=(1.5, None)), pg.NumericVectorPattern(4, lb=1.5)),
    (ParameterSpec("a", (), C.UPPER, bounds=(None, -2.0)), pg.NumericScalarPattern(ub=-2.0)),
    (ParameterSpec("a", (3,), C.LOWER_UPPER, bounds=(-1.0, 4.0)), pg.NumericVectorPattern(3, lb=-1.0, ub=4.0)),
    (ParameterSpec("a", (4,), C.ORDERED), pg.OrderedVectorPattern(4)),
    (ParameterSpec("a", (3,), C.POSITIVE_ORDERED), pg.OrderedVectorPattern(3, positive=True)),
    (ParameterSpec("a", (5,), C.SIMPLEX), pg.SimplexArrayPattern(5)),
    (ParameterSpec("a", (3, 4), C.SIMPLEX), pg.SimplexArrayPattern(4, array_shape=(3,))),
    (ParameterSpec("a", (4, 4), C.CHOLESKY_FACTOR_CORR), pg.CholeskyCorrPattern(4)),
    (ParameterSpec("a", (3, 3), C.CHOLESKY_FACTOR_COV), pg.CholeskyCovPattern(3)),
    (ParameterSpec("a", (3, 3), C.COV_MATRIX), pg.PSDSymmetricMatrixPattern(3)),
]


@pytest.mark.parametrize("spec,pattern", CASES, ids=[f"{s.constraint.name}{s.shape}" for s, _ in CASES])
def test_same_transform(spec, pattern):
    ms = ModelSpec((spec,))
    assert pattern.free_length() == ms.free_dim
    for k in jax.random.split(jax.random.PRNGKey(0), 5):
        x = 0.7 * jax.random.normal(k, (ms.free_dim,))
        old, old_logdet = ms.constrain(x)
        new, new_logdet = pattern.free_to_folded_with_log_det(x)
        np.testing.assert_allclose(new, old["a"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(new_logdet, old_logdet, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(pattern.flatten(new, to_free=True), ms.unconstrain(old)[0], rtol=1e-8, atol=1e-8)
