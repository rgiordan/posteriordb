"""posteriordb posteriors as modelgami `Model`s, ready to use:

    from posteriordb_jax import catalog
    model = catalog.load("kidiq-kidscore_momiq")       # data loaded and converted
    for name in catalog.list_models(exchangeable=True):  # run a method across models
        run_method(catalog.load(name))
    draws = catalog.reference_draws("kidiq-kidscore_momiq")  # gold-standard posterior draws

Each module here translates one Stan model and lists the posteriors (data
sets) it serves. This is the new-style counterpart of `posteriordb_jax.models`;
models move here as they are converted.
"""

from __future__ import annotations

import importlib
from typing import Dict, List, Optional

import jax.numpy as jnp

import modelgami as mg

from ..io import load_data, load_posterior_info, load_reference_draws
from ..stan_names import folded_draws_from_posteriordb

_MODULES = [
    "2pl_latent_reg_irt",
    "accel_gp",
    "accel_splines",
    "arK",
    "arma11",
    "blr",
    "bones_model",
    "bym2_offset_only",
    "covid19imperial_v2",
    "covid19imperial_v3",
    "diamonds",
    "dogs",
    "dogs_hierarchical",
    "dogs_log",
    "dogs_nonhierarchical",
    "dugongs_model",
    "earn_height",
    "eight_schools_centered",
    "eight_schools_noncentered",
    "election88_full",
    "garch11",
    "GLM_Binomial_model",
    "GLM_Poisson_model",
    "GLMM1_model",
    "GLMM_Poisson_model",
    "gp_pois_regr",
    "gp_regr",
    "gpcm_latent_reg_irt",
    "grsm_latent_reg_irt",
    "hier_2pl",
    "hierarchical_gp",
    "hmm_drive_0",
    "hmm_drive_1",
    "hmm_example",
    "hmm_gaussian",
    "iohmm_reg",
    "irt_2pl",
    "kidscore_interaction",
    "kidscore_interaction_c",
    "kidscore_interaction_c2",
    "kidscore_interaction_z",
    "kidscore_mom_work",
    "kidscore_momhs",
    "kidscore_momhsiq",
    "kidscore_momiq",
    "kilpisjarvi",
    "kronecker_gp",
    "ldaK2",
    "ldaK5",
    "log10earn_height",
    "logearn_height",
    "logearn_height_male",
    "logearn_interaction",
    "logearn_interaction_z",
    "logearn_logheight_male",
    "logistic_regression_rhs",
    "logmesquite",
    "logmesquite_logva",
    "logmesquite_logvas",
    "logmesquite_logvash",
    "logmesquite_logvolume",
    "losscurve_sislob",
    "low_dim_gauss_mix",
    "low_dim_gauss_mix_collapse",
    "lsat_model",
    "M0_model",
    "Mb_model",
    "mesquite",
    "Mh_model",
    "Mt_model",
    "Mtbh_model",
    "Mth_model",
    "multi_occupancy",
    "nes",
    "nes_logit_model",
    "nn_rbm1bJ10",
    "nn_rbm1bJ100",
    "normal_mixture",
    "normal_mixture_k",
    "pilots",
    "prophet",
    "radon_county",
    "radon_county_intercept",
    "radon_hierarchical_intercept_centered",
    "radon_hierarchical_intercept_noncentered",
    "radon_partially_pooled_centered",
    "radon_partially_pooled_noncentered",
    "radon_pooled",
    "radon_variable_intercept_centered",
    "radon_variable_intercept_noncentered",
    "radon_variable_intercept_slope_centered",
    "radon_variable_intercept_slope_noncentered",
    "radon_variable_slope_centered",
    "radon_variable_slope_noncentered",
    "Rate_1_model",
    "Rate_2_model",
    "Rate_3_model",
    "Rate_4_model",
    "Rate_5_model",
    "rats_model",
    "seeds_centered_model",
    "seeds_model",
    "seeds_stanified_model",
    "sesame_one_pred_a",
    "state_space_stochastic_level_stochastic_seasonal",
    "surgical_model",
    "Survey_model",
    "wells_daae_c_model",
    "wells_dae_c_model",
    "wells_dae_inter_model",
    "wells_dae_model",
    "wells_dist",
    "wells_dist100_model",
    "wells_dist100ars_model",
    "wells_interaction_c_model",
    "wells_interaction_model",
]


def _registry() -> Dict[str, object]:
    reg = {}
    for m in _MODULES:
        mod = importlib.import_module(f".{m}", __name__)
        for posterior in mod.POSTERIORS:
            reg[posterior] = mod
    return reg


def module_for(posterior_name: str):
    reg = _registry()
    if posterior_name not in reg:
        raise KeyError(f"{posterior_name!r} is not converted yet; available: {sorted(reg)}")
    return reg[posterior_name]


def load(posterior_name: str):
    """The `modelgami.Model` for a posteriordb posterior (e.g.
    "eight_schools-eight_schools_noncentered"), with its data loaded. Data
    lists become float64 arrays; each module casts its integer fields."""
    mod = module_for(posterior_name)
    info = load_posterior_info(posterior_name)
    raw = {k: (jnp.asarray(v, dtype=jnp.float64) if isinstance(v, list) else v)
           for k, v in load_data(info["data_name"]).items()}
    model = mod.build(raw)
    model.name = posterior_name
    return model


def list_models(exchangeable: Optional[bool] = None) -> List[str]:
    """Posterior names, optionally only those whose model is (or is not) a
    `modelgami.ExchangeableModel`."""
    names = sorted(_registry())
    if exchangeable is None:
        return names
    return [name for name in names if isinstance(load(name), mg.ExchangeableModel) == exchangeable]


def reference_draws(posterior_name: str, as_arviz: bool = False):
    """posteriordb's reference posterior draws, as a dict of folded arrays
    with leading (chain, draw) axes (or an ArviZ object with `as_arviz`).
    Raises FileNotFoundError when posteriordb has none for this posterior."""
    model = load(posterior_name)
    draws = folded_draws_from_posteriordb(model.params_pattern, load_reference_draws(posterior_name))
    return model.to_arviz(draws) if as_arviz else draws
