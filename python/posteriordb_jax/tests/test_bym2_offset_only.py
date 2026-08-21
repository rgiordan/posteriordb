import jax.numpy as jnp

import posteriordb_jax  # noqa: F401
from posteriordb_jax.io import load_data, stan_file_path
from posteriordb_jax.models.bym2_offset_only import Bym2OffsetOnly
from posteriordb_jax.testing import StanComparisonHarness, assert_matches_stan


def _load_model():
    raw = load_data("traffic_accident_nyc")
    data = {
        "N": raw["N"],
        "N_edges": raw["N_edges"],
        "node1": raw["node1"],
        "node2": raw["node2"],
        "y": raw["y"],
        "E": jnp.asarray(raw["E"], dtype=jnp.float64),
        "scaling_factor": raw["scaling_factor"],
    }
    return Bym2OffsetOnly(data), data


def test_free_dim_matches_stan():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("bym2_offset_only"), data)
    assert harness.param_unc_num() == model.model_spec.free_dim


def test_matches_bridgestan_log_density_and_grad():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("bym2_offset_only"), data)
    # N=1921 -> free_dim ~= 3845; Hessian comparison skipped for the same
    # size reason as test_logistic_regression_rhs.py.
    report = assert_matches_stan(model, harness, n_synthetic=5, seed=0, check_hessian=False, scale=0.3)
    report.assert_within(atol=1e-3)
