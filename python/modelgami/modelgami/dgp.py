"""`DGP`: a data-generating process, i.e. a log density plus (optionally) a
way to draw from it. A `Model` exposes its prior and likelihood as DGPs
(`model.prior`, `model.likelihood`), so code that only cares about one
piece can take it on its own."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class DGP:
    log_density: Callable
    sample: Optional[Callable] = None
    pointwise: Optional[Callable] = None  # exchangeable likelihoods only

    @property
    def can_sample(self) -> bool:
        return self.sample is not None

    @property
    def exchangeable(self) -> bool:
        return self.pointwise is not None
