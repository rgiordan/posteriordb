"""Stan-free tests for StatisticalModel: grad/hessian/hvp vs finite
differences, batching == looped single-draw calls, and grad-wrt-data vs
finite differences -- using a hand-rolled toy model (no Stan involved)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import posteriordb_jax  # noqa: F401
from posteriordb_jax.constraints import ConstraintType, ParameterSpec
from posteriordb_jax.model import StatisticalModel

jax.config.update("jax_enable_x64", True)


class ToyGaussianModel(StatisticalModel):
    """mu ~ Normal(0, prior_scale); sigma > 0 (flat prior);
    y_i ~ Normal(mu, sigma) iid."""

    def param_specs(self, data):
        return [
            ParameterSpec("mu", (), ConstraintType.UNCONSTRAINED),
            ParameterSpec("sigma", (), ConstraintType.LOWER, bounds=(0.0, None)),
        ]

    def log_density(self, params, data):
        mu = params["mu"]
        sigma = params["sigma"]
        y = data["y"]
        prior_scale = data["prior_scale"]
        lp = -0.5 * jnp.sum(((y - mu) / sigma) ** 2) - y.shape[0] * jnp.log(sigma)
        lp = lp - 0.5 * (mu / prior_scale) ** 2
        return lp


@pytest.fixture
def toy():
    data = {"y": jnp.asarray([1.0, 2.0, -0.5, 0.3, 1.8]), "prior_scale": jnp.asarray(2.0)}
    return ToyGaussianModel(data)


def _finite_diff_grad(f, x, eps=1e-6):
    x = np.asarray(x)
    g = np.zeros_like(x)
    for i in range(x.size):
        dx = np.zeros_like(x)
        dx[i] = eps
        g[i] = (f(x + dx) - f(x - dx)) / (2 * eps)
    return g


def test_log_prob_unconstrained_matches_constrained_at_matching_point(toy):
    x = jnp.array([0.3, -1.0])  # mu unconstrained, sigma unconstrained
    params = toy.constrain(x)
    y = jnp.array([params["mu"], params["sigma"]])
    lp_x = toy.log_prob_unconstrained(x)
    lp_y_plus_jac = toy.log_prob_constrained(y) + toy.model_spec.specs[1].bijector.forward_log_det_jacobian(
        x[1:2], event_ndims=1
    )
    np.testing.assert_allclose(lp_x, lp_y_plus_jac, atol=1e-8)


def test_grad_unconstrained_vs_finite_diff(toy):
    x = jnp.array([0.3, -1.0])
    analytic = np.asarray(toy.grad_unconstrained(x))
    fd = _finite_diff_grad(lambda x_: float(toy.log_prob_unconstrained(jnp.asarray(x_))), x)
    np.testing.assert_allclose(analytic, fd, atol=1e-5)


def test_hessian_unconstrained_vs_finite_diff(toy):
    x = np.array([0.3, -1.0])
    eps = 1e-4
    n = 2
    h_fd = np.zeros((n, n))
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        gp = np.asarray(toy.grad_unconstrained(jnp.asarray(x + dx)))
        gm = np.asarray(toy.grad_unconstrained(jnp.asarray(x - dx)))
        h_fd[:, i] = (gp - gm) / (2 * eps)
    h_analytic = np.asarray(toy.hessian_unconstrained(jnp.asarray(x)))
    np.testing.assert_allclose(h_analytic, h_fd, atol=1e-4)


def test_hvp_matches_hessian_matvec(toy):
    x = jnp.array([0.3, -1.0])
    v = jnp.array([0.5, -0.2])
    h = toy.hessian_unconstrained(x)
    expected = h @ v
    hvp = toy.hvp_unconstrained(x, v)
    np.testing.assert_allclose(hvp, expected, atol=1e-8)


def test_batch_matches_loop(toy):
    rng = np.random.default_rng(0)
    xs = jnp.asarray(rng.normal(size=(6, 2)))
    lp_batch = toy.log_prob_unconstrained_batch(xs)
    grad_batch = toy.grad_unconstrained_batch(xs)
    hess_batch = toy.hessian_unconstrained_batch(xs)
    for i in range(6):
        np.testing.assert_allclose(lp_batch[i], toy.log_prob_unconstrained(xs[i]), atol=1e-10)
        np.testing.assert_allclose(grad_batch[i], toy.grad_unconstrained(xs[i]), atol=1e-10)
        np.testing.assert_allclose(hess_batch[i], toy.hessian_unconstrained(xs[i]), atol=1e-10)


def test_hvp_batch_matches_loop(toy):
    rng = np.random.default_rng(1)
    xs = jnp.asarray(rng.normal(size=(4, 2)))
    vs = jnp.asarray(rng.normal(size=(4, 2)))
    hvp_batch = toy.hvp_unconstrained_batch(xs, vs)
    for i in range(4):
        np.testing.assert_allclose(hvp_batch[i], toy.hvp_unconstrained(xs[i], vs[i]), atol=1e-10)


def test_grad_wrt_data_vs_finite_diff(toy):
    x = jnp.array([0.3, -1.0])

    def f(prior_scale):
        d = {"y": toy.data["y"], "prior_scale": jnp.asarray(prior_scale)}
        return float(toy.log_prob_unconstrained(x, data=d))

    eps = 1e-6
    ps = float(toy.data["prior_scale"])
    fd = (f(ps + eps) - f(ps - eps)) / (2 * eps)

    grad_data = toy.grad_wrt_data(x)
    np.testing.assert_allclose(grad_data["prior_scale"], fd, atol=1e-5)


def test_grad_wrt_data_y_vs_finite_diff(toy):
    x = jnp.array([0.3, -1.0])
    grad_data = toy.grad_wrt_data(x)
    grad_y = np.asarray(grad_data["y"])

    y0 = np.asarray(toy.data["y"])
    eps = 1e-6
    for i in range(len(y0)):
        dy = np.zeros_like(y0)
        dy[i] = eps
        d_plus = {"y": jnp.asarray(y0 + dy), "prior_scale": toy.data["prior_scale"]}
        d_minus = {"y": jnp.asarray(y0 - dy), "prior_scale": toy.data["prior_scale"]}
        fd = (
            float(toy.log_prob_unconstrained(x, data=d_plus))
            - float(toy.log_prob_unconstrained(x, data=d_minus))
        ) / (2 * eps)
        np.testing.assert_allclose(grad_y[i], fd, atol=1e-5)


def test_constrain_unconstrain_and_posteriordb_dict_roundtrip(toy):
    x = jnp.array([0.3, -1.0])
    params = toy.constrain(x)
    x_back = toy.unconstrain(params)
    np.testing.assert_allclose(x_back, x, atol=1e-8)

    pdb_dict = toy.to_posteriordb_dict(params)
    assert set(pdb_dict.keys()) == {"mu", "sigma"}
    params_back = toy.from_posteriordb_dict(pdb_dict)
    np.testing.assert_allclose(params_back["mu"], params["mu"], atol=1e-8)
    np.testing.assert_allclose(params_back["sigma"], params["sigma"], atol=1e-8)
