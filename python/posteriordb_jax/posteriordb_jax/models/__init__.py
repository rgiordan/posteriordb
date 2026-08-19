"""Registry of hand-converted Stan models, keyed by posteriordb model name
(matching `posterior_database/models/stan/<name>.stan`)."""

from .dogs_nonhierarchical import DogsNonhierarchical
from .eight_schools_noncentered import EightSchoolsNoncentered
from .hmm_example import HmmExample

MODEL_REGISTRY = {
    "eight_schools_noncentered": EightSchoolsNoncentered,
    "hmm_example": HmmExample,
    "dogs_nonhierarchical": DogsNonhierarchical,
}
