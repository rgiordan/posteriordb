"""modelgami: a common interface for statistical models in JAX.

A model is a paragami pattern for its parameters, a prior and a likelihood
(both data-generating processes, `DGP`s), its data and hyperparameters.
From these modelgami builds log densities in folded, flat or free form,
and, for exchangeable data, pointwise and weighted likelihoods.
"""

from .dgp import DGP, ExchangeableDGP, exchangeable_dgp_from_distributions, n_obs
from .model import ExchangeableModel, Model, hvp
from .numpyro_adapter import from_numpyro
from .stan import StanModel, from_bridgestan

__version__ = "0.1.0.dev0"
