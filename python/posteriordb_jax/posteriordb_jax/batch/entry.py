"""`Entry`: one posteriordb posterior, as an analysis sees it."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, Iterable, List, Optional

import modelgami as mg

from .. import catalog
from ..io import load_posterior_info


@dataclass(frozen=True)
class Entry:
    """One posteriordb posterior. The model and posteriordb's metadata are
    loaded on first use, so making entries is cheap."""

    name: str   # posteriordb's posterior name, "<data>-<model>"

    @cached_property
    def model(self) -> mg.Model:
        """The catalog's modelgami model, with posteriordb's data."""
        return catalog.load(self.name)

    @cached_property
    def info(self) -> Dict[str, Any]:
        """posteriordb's posterior metadata (keywords, references, ...)."""
        return load_posterior_info(self.name)

    @property
    def stan_model(self) -> str:
        return self.info["model_name"]

    @property
    def data_name(self) -> str:
        return self.info["data_name"]

    @property
    def is_exchangeable(self) -> bool:
        """Whether the data are exchangeable observations, so that bootstrap
        weights make sense."""
        return isinstance(self.model, mg.ExchangeableModel)

    @property
    def has_reference_draws(self) -> bool:
        return self.info.get("reference_posterior_name") is not None

    def load_reference_draws(self) -> Optional[Dict[str, Any]]:
        """posteriordb's reference posterior draws as folded arrays with
        leading (chain, draw) axes, or None when there are none."""
        return catalog.load_reference_draws(self.name) if self.has_reference_draws else None


def select_entries(names: Optional[Iterable[str]] = None, *, exchangeable: Optional[bool] = None,
            with_reference_draws: Optional[bool] = None) -> List[Entry]:
    """The catalog's posteriors (or `names`), optionally only those whose
    data are (or are not) exchangeable, or that have (or lack) reference
    draws. The exchangeability filter loads every model."""
    selected = [Entry(name) for name in (catalog.list_models() if names is None else names)]
    if with_reference_draws is not None:
        selected = [e for e in selected if e.has_reference_draws == with_reference_draws]
    if exchangeable is not None:
        selected = [e for e in selected if e.is_exchangeable == exchangeable]
    return selected
