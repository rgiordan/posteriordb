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
from typing import Dict, List

from ..io import load_data, load_posterior_info, load_reference_draws
from ..stan_names import folded_draws_from_posteriordb

_MODULES = [
    "blr",
    "eight_schools_centered",
    "eight_schools_noncentered",
    "kidscore_momiq",
    "kilpisjarvi",
    "logearn_height",
    "low_dim_gauss_mix",
    "normal_mixture",
    "wells_dist",
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
    "eight_schools-eight_schools_noncentered"), with its data loaded."""
    mod = module_for(posterior_name)
    info = load_posterior_info(posterior_name)
    model = mod.build(load_data(info["data_name"]))
    model.name = posterior_name
    return model


def list_models(**filters) -> List[str]:
    """Posterior names whose `Model.capabilities` match every filter, e.g.
    `list_models(exchangeable=True, has_prior=True)`."""
    names = sorted(_registry())
    if not filters:
        return names
    out = []
    for name in names:
        caps = load(name).capabilities
        if all(caps.get(k) == v for k, v in filters.items()):
            out.append(name)
    return out


def reference_draws(posterior_name: str, as_arviz: bool = False):
    """posteriordb's reference posterior draws, as a dict of folded arrays
    with leading (chain, draw) axes (or an ArviZ object with `as_arviz`).
    Raises FileNotFoundError when posteriordb has none for this posterior."""
    model = load(posterior_name)
    draws = folded_draws_from_posteriordb(model.params_pattern, load_reference_draws(posterior_name))
    return model.to_arviz(draws) if as_arviz else draws
