"""modelgami: a common interface for statistical models in JAX.

Write a model's parameter pattern, log prior and log likelihood once, in
terms of dicts of arrays; get log densities in folded/flat/free form,
exchangeable-data tools (pointwise, weighted and subset likelihoods),
validation, simulation and ArviZ export generated for you.
"""

from .dgp import DGP
from .model import Model, hvp
from .numpyro_adapter import from_numpyro
from .stan import from_bridgestan
from .validation import ValidationError

__version__ = "0.1.0.dev0"
