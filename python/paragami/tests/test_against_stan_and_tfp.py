"""Every `bijector="stan"` transform against Stan itself (through BridgeStan),
and every `bijector="tfp"` transform against TensorFlow Probability.

Both are optional: the Stan tests compile a model (slow; marked `stan`) and
are skipped without BridgeStan, and the TFP tests are skipped when TFP
cannot be imported (it needs jax <= 0.5)."""

import json
import math
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

try:
    from tensorflow_probability.substrates import jax as tfp
except Exception:  # not installed, or incompatible with this JAX version
    tfp = None

import paragami as pg

jax.config.update("jax_enable_x64", True)

STAN_PROGRAM = textwrap.dedent("""
    data {
      matrix[2, 2] m_offset;
      matrix[2, 2] m_multiplier;
    }
    parameters {
      vector<lower=1>[3] a;
      vector<offset=1, multiplier=2>[2] b;
      unit_vector[3] c;
      sum_to_zero_vector[4] d;
      sum_to_zero_matrix[3, 4] e;
      column_stochastic_matrix[3, 2] f;
      row_stochastic_matrix[3, 2] g;
      corr_matrix[4] h;
      cholesky_factor_corr[4] i;
      cov_matrix[3] j;
      simplex[4] k;
      array[2] simplex[3] l;
      matrix<offset=m_offset, multiplier=m_multiplier>[2, 2] m;
      positive_ordered[3] n;
      ordered[3] o;
      real<lower=0, upper=2> p;
      cholesky_factor_cov[3] q;
      real<upper=-1> r;
    }
""")
STAN_DATA = {"m_offset": [[1.0, 2.0], [3.0, 4.0]], "m_multiplier": [[0.5, 1.0], [2.0, 3.0]]}


def stan_pattern():
    """The same parameters as STAN_PROGRAM, in the same order."""
    return pg.PatternDict(
        a=pg.NumericVectorPattern(3, lb=1.0),
        b=pg.NumericVectorPattern(2, offset=1.0, multiplier=2.0),
        c=pg.UnitVectorPattern(3),
        d=pg.SumToZeroVectorPattern(4),
        e=pg.SumToZeroMatrixPattern(3, 4),
        f=pg.StochasticMatrixPattern(3, 2, kind="column"),
        g=pg.StochasticMatrixPattern(3, 2, kind="row"),
        h=pg.CorrMatrixPattern(4),
        i=pg.CholeskyCorrPattern(4),
        j=pg.PSDSymmetricMatrixPattern(3),
        k=pg.SimplexArrayPattern(4),
        l=pg.SimplexArrayPattern(3, array_shape=(2,)),
        m=pg.NumericArrayPattern((2, 2), offset=STAN_DATA["m_offset"], multiplier=STAN_DATA["m_multiplier"],
                                 order="F"),
        n=pg.OrderedVectorPattern(3, positive=True),
        o=pg.OrderedVectorPattern(3),
        p=pg.NumericScalarPattern(lb=0.0, ub=2.0),
        q=pg.CholeskyCovPattern(3),
        r=pg.NumericScalarPattern(ub=-1.0),
    )


def stan_layout(value):
    """A folded value in Stan's constrained order: column-major, and arrays
    of vectors element by element."""
    value = np.asarray(value)
    return value.reshape(-1, order="F") if value.ndim == 2 else value.reshape(-1)


@pytest.fixture(scope="module")
def stan_model(tmp_path_factory):
    bs = pytest.importorskip("bridgestan")
    path = tmp_path_factory.mktemp("stan")
    (path / "transforms.stan").write_text(STAN_PROGRAM)
    (path / "transforms.json").write_text(json.dumps(STAN_DATA))
    return bs.StanModel(str(path / "transforms.stan"), data=str(path / "transforms.json"))


@pytest.mark.stan
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_stan_transforms(stan_model, seed):
    pattern = stan_pattern()
    assert pattern.free_length() == stan_model.param_unc_num()
    u = np.random.default_rng(seed).normal(size=pattern.free_length()) * 0.7
    theta = stan_model.param_constrain(u)
    folded = pattern.free_to_folded(u)
    offset = 0
    for name, p in pattern.items():
        n = p.flat_length()
        np.testing.assert_allclose(stan_layout(folded[name]), theta[offset:offset + n], atol=1e-12,
                                   err_msg=f"value of {name}")
        offset += n
    # With no model block, the log density is the sum of the log Jacobians.
    stan_log_det = stan_model.log_density(u, jacobian=True) - stan_model.log_density(u, jacobian=False)
    np.testing.assert_allclose(pattern.free_to_folded_log_abs_det(u), stan_log_det, rtol=1e-12)
    np.testing.assert_allclose(pattern.folded_to_free(folded), stan_model.param_unconstrain(theta), atol=1e-10)


@pytest.fixture(scope="module")
def tfb():
    if tfp is None:  # TFP is optional and needs jax <= 0.5
        pytest.skip("TensorFlow Probability unavailable")
    return tfp.bijectors


def _tfp_cases(tfb):
    one, two, shift = jnp.asarray(1.0), jnp.asarray(2.0), jnp.asarray(pg.matrix.TFP_DIAG_SHIFT)
    return {
        "lb": (pg.NumericVectorPattern(3, lb=1.0, bijector="tfp"), tfb.Chain([tfb.Shift(one), tfb.Softplus()])),
        "ub": (pg.NumericVectorPattern(3, ub=2.0, bijector="tfp"),
               tfb.Chain([tfb.Shift(two), tfb.Scale(-one), tfb.Softplus()])),
        "lub": (pg.NumericVectorPattern(3, lb=-1.0, ub=2.0, bijector="tfp"), tfb.Sigmoid(low=-one, high=two)),
        "ordered": (pg.OrderedVectorPattern(4), tfb.Ascending()),
        "chol_cov": (pg.CholeskyCovPattern(4, bijector="tfp"), tfb.FillScaleTriL(diag_shift=shift)),
        "cov": (pg.PSDSymmetricMatrixPattern(4, bijector="tfp"),
                tfb.Chain([tfb.CholeskyOuterProduct(), tfb.FillScaleTriL(diag_shift=shift)])),
        "chol_corr": (pg.CholeskyCorrPattern(4, bijector="tfp"), tfb.CorrelationCholesky()),
        "corr": (pg.CorrMatrixPattern(4, bijector="tfp"),
                 tfb.Chain([tfb.CholeskyOuterProduct(), tfb.CorrelationCholesky()])),
    }


@pytest.mark.parametrize("case", ["lb", "ub", "lub", "ordered", "chol_cov", "cov", "chol_corr", "corr"])
def test_tfp_transforms(tfb, case):
    pattern, bijector = _tfp_cases(tfb)[case]
    u = jnp.asarray(np.random.default_rng(0).normal(size=pattern.free_length()) * 0.7)
    np.testing.assert_allclose(pattern.free_to_folded(u), bijector.forward(u), atol=1e-12)
    np.testing.assert_allclose(pattern.free_to_folded_log_abs_det(u),
                               bijector.forward_log_det_jacobian(u, event_ndims=1), atol=1e-12)
    np.testing.assert_allclose(pattern.folded_to_free(pattern.free_to_folded(u)), u, atol=1e-10)


def test_tfp_softmax_centered():
    """TFP's SoftmaxCentered, from its definition (log det as in TFP's
    `_forward_log_det_jacobian`), since constructing the bijector fails
    with some numpy versions."""
    pattern = pg.SimplexArrayPattern(4, bijector="tfp")
    u = jnp.asarray([0.3, -0.5, 1.2])
    np.testing.assert_allclose(pattern.free_to_folded(u), jax.nn.softmax(jnp.asarray([0.3, -0.5, 1.2, 0.0])))
    expected = 0.5 * math.log(4.0) + jnp.sum(u) - 4 * jax.nn.softplus(jax.scipy.special.logsumexp(u))
    np.testing.assert_allclose(pattern.free_to_folded_log_abs_det(u), expected, rtol=1e-12)
