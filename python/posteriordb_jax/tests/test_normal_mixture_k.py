import jax.numpy as jnp

import posteriordb_jax  # noqa: F401
from posteriordb_jax.io import load_data, stan_file_path
from posteriordb_jax.models.normal_mixture_k import NormalMixtureK
from posteriordb_jax.testing import StanComparisonHarness, assert_matches_stan


def _load_model():
    raw = load_data("normal_5")
    data = {
        "K": raw["K"],
        "N": raw["N"],
        "y": jnp.asarray(raw["y"], dtype=jnp.float64),
    }
    return NormalMixtureK(data), data


def test_param_specs_order_matches_stan():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("normal_mixture_k"), data)
    assert harness.param_unc_num() == model.model_spec.free_dim
    assert harness.param_unc_names() == [
        "theta.1", "theta.2", "theta.3", "theta.4",
        "mu.1", "mu.2", "mu.3", "mu.4", "mu.5",
        "sigma.1", "sigma.2", "sigma.3", "sigma.4", "sigma.5",
    ]


def test_matches_bridgestan_log_density_grad_hessian():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("normal_mixture_k"), data)
    # A mixture log-density is only piecewise-smooth in general; keep
    # synthetic draws modest to avoid a pathological (near-degenerate
    # component) region, as in test_low_dim_gauss_mix.py.
    report = assert_matches_stan(model, harness, n_synthetic=20, seed=0, check_hessian=True, scale=0.3)
    report.assert_within(atol=1e-4, atol_hessian=1e-2)
