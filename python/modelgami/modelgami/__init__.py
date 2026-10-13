"""modelgami: a common interface for statistical models in JAX.

A model is a prior and a likelihood (both data-generating processes,
`DGP`s), a paragami pattern for its parameters, its data and its
hyperparameters. The model combines the two into a log joint density and
draws from the joint and marginal distributions; for exchangeable data it
adds single-observation, weighted and subset likelihoods.
"""

from .dgp import (
    DGP,
    ExchangeableDGP,
    count_obs,
    draw_np_bootstrap_weights,
    exchangeable_dgp_from_distributions,
    folded_dgp,
    free_dgp,
)
from .model import ExchangeableModel, Model, make_hvp
from .numpyro_adapter import from_numpyro
from .stan import StanModel, from_bridgestan

__version__ = "0.1.0.dev0"
