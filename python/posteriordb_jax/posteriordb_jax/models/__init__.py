"""Registry of hand-converted Stan models, keyed by posteriordb model name
(matching `posterior_database/models/stan/<name>.stan`)."""

from .arK import ArK
from .arma11 import Arma11
from .blr import Blr
from .bym2_offset_only import Bym2OffsetOnly
from .dogs import Dogs
from .dogs_log import DogsLog
from .dogs_nonhierarchical import DogsNonhierarchical
from .diamonds import Diamonds
from .dugongs_model import DugongsModel
from .earn_height import EarnHeight
from .eight_schools_centered import EightSchoolsCentered
from .eight_schools_noncentered import EightSchoolsNoncentered
from .GLM_Poisson_model import GLMPoissonModel
from .GLM_Binomial_model import GLMBinomialModel
from .GLMM1_model import GLMM1Model
from .GLMM_Poisson_model import GLMMPoissonModel
from .gp_regr import GpRegr
from .hier_2pl import Hier2pl
from .hmm_example import HmmExample
from .irt_2pl import Irt2pl
from .kilpisjarvi import Kilpisjarvi
from .kidscore_interaction import KidscoreInteraction
from .ldaK2 import LdaK2
from .ldaK5 import LdaK5
from .kidscore_interaction_c import KidscoreInteractionC
from .kidscore_interaction_c2 import KidscoreInteractionC2
from .kidscore_interaction_z import KidscoreInteractionZ
from .kidscore_mom_work import KidscoreMomWork
from .kidscore_momhs import KidscoreMomhs
from .kidscore_momhsiq import KidscoreMomhsiq
from .kidscore_momiq import KidscoreMomiq
from .log10earn_height import Log10earnHeight
from .logearn_height import LogearnHeight
from .logearn_height_male import LogearnHeightMale
from .logearn_interaction import LogearnInteraction
from .logearn_interaction_z import LogearnInteractionZ
from .logearn_logheight_male import LogearnLogheightMale
from .logmesquite import Logmesquite
from .logmesquite_logva import LogmesquiteLogva
from .logmesquite_logvas import LogmesquiteLogvas
from .logmesquite_logvash import LogmesquiteLogvash
from .logistic_regression_rhs import LogisticRegressionRhs
from .logmesquite_logvolume import LogmesquiteLogvolume
from .low_dim_gauss_mix import LowDimGaussMix
from .low_dim_gauss_mix_collapse import LowDimGaussMixCollapse
from .lsat_model import LsatModel
from .M0_model import M0Model
from .mesquite import Mesquite
from .Mt_model import MtModel
from .nes import Nes
from .nes_logit_model import NesLogitModel
from .normal_mixture import NormalMixture
from .normal_mixture_k import NormalMixtureK
from .pilots import Pilots
from .radon_county import RadonCounty
from .radon_county_intercept import RadonCountyIntercept
from .radon_hierarchical_intercept_centered import RadonHierarchicalInterceptCentered
from .radon_hierarchical_intercept_noncentered import RadonHierarchicalInterceptNoncentered
from .radon_partially_pooled_centered import RadonPartiallyPooledCentered
from .radon_partially_pooled_noncentered import RadonPartiallyPooledNoncentered
from .radon_pooled import RadonPooled
from .radon_variable_intercept_centered import RadonVariableInterceptCentered
from .radon_variable_intercept_noncentered import RadonVariableInterceptNoncentered
from .radon_variable_intercept_slope_centered import RadonVariableInterceptSlopeCentered
from .radon_variable_intercept_slope_noncentered import RadonVariableInterceptSlopeNoncentered
from .radon_variable_slope_centered import RadonVariableSlopeCentered
from .radon_variable_slope_noncentered import RadonVariableSlopeNoncentered
from .rats_model import RatsModel
from .Rate_1_model import Rate1Model
from .Rate_2_model import Rate2Model
from .Rate_3_model import Rate3Model
from .Rate_4_model import Rate4Model
from .Rate_5_model import Rate5Model
from .seeds_centered_model import SeedsCenteredModel
from .seeds_model import SeedsModel
from .seeds_stanified_model import SeedsStanifiedModel
from .sesame_one_pred_a import SesameOnePredA
from .surgical_model import SurgicalModel
from .wells_daae_c_model import WellsDaaeCModel
from .wells_dae_c_model import WellsDaeCModel
from .wells_dae_inter_model import WellsDaeInterModel
from .wells_dae_model import WellsDaeModel
from .wells_dist import WellsDist
from .wells_dist100_model import WellsDist100Model
from .wells_dist100ars_model import WellsDist100arsModel
from .wells_interaction_c_model import WellsInteractionCModel
from .wells_interaction_model import WellsInteractionModel

MODEL_REGISTRY = {
    "eight_schools_noncentered": EightSchoolsNoncentered,
    "eight_schools_centered": EightSchoolsCentered,
    "hmm_example": HmmExample,
    "dogs_nonhierarchical": DogsNonhierarchical,
    "blr": Blr,
    "mesquite": Mesquite,
    "nes_logit_model": NesLogitModel,
    "low_dim_gauss_mix": LowDimGaussMix,
    "arK": ArK,
    "wells_dist": WellsDist,
    "earn_height": EarnHeight,
    "sesame_one_pred_a": SesameOnePredA,
    "kidscore_momhs": KidscoreMomhs,
    "kidscore_momiq": KidscoreMomiq,
    "kidscore_momhsiq": KidscoreMomhsiq,
    "Rate_1_model": Rate1Model,
    "Rate_2_model": Rate2Model,
    "Rate_3_model": Rate3Model,
    "Rate_4_model": Rate4Model,
    "Rate_5_model": Rate5Model,
    "wells_dist100_model": WellsDist100Model,
    "wells_dist100ars_model": WellsDist100arsModel,
    "wells_dae_model": WellsDaeModel,
    "wells_dae_c_model": WellsDaeCModel,
    "wells_interaction_model": WellsInteractionModel,
    "wells_interaction_c_model": WellsInteractionCModel,
    "wells_daae_c_model": WellsDaaeCModel,
    "wells_dae_inter_model": WellsDaeInterModel,
    "logearn_height": LogearnHeight,
    "log10earn_height": Log10earnHeight,
    "logearn_height_male": LogearnHeightMale,
    "logearn_logheight_male": LogearnLogheightMale,
    "logearn_interaction": LogearnInteraction,
    "logearn_interaction_z": LogearnInteractionZ,
    "kidscore_interaction": KidscoreInteraction,
    "kidscore_interaction_c": KidscoreInteractionC,
    "kidscore_interaction_c2": KidscoreInteractionC2,
    "kidscore_interaction_z": KidscoreInteractionZ,
    "kidscore_mom_work": KidscoreMomWork,
    "logmesquite": Logmesquite,
    "logmesquite_logva": LogmesquiteLogva,
    "logmesquite_logvas": LogmesquiteLogvas,
    "logmesquite_logvash": LogmesquiteLogvash,
    "logmesquite_logvolume": LogmesquiteLogvolume,
    "normal_mixture_k": NormalMixtureK,
    "normal_mixture": NormalMixture,
    "kilpisjarvi": Kilpisjarvi,
    "dugongs_model": DugongsModel,
    "radon_pooled": RadonPooled,
    "radon_county": RadonCounty,
    "radon_county_intercept": RadonCountyIntercept,
    "ldaK2": LdaK2,
    "ldaK5": LdaK5,
    "low_dim_gauss_mix_collapse": LowDimGaussMixCollapse,
    "radon_partially_pooled_centered": RadonPartiallyPooledCentered,
    "radon_partially_pooled_noncentered": RadonPartiallyPooledNoncentered,
    "radon_variable_intercept_centered": RadonVariableInterceptCentered,
    "radon_variable_intercept_noncentered": RadonVariableInterceptNoncentered,
    "radon_variable_slope_centered": RadonVariableSlopeCentered,
    "radon_variable_slope_noncentered": RadonVariableSlopeNoncentered,
    "radon_variable_intercept_slope_centered": RadonVariableInterceptSlopeCentered,
    "radon_variable_intercept_slope_noncentered": RadonVariableInterceptSlopeNoncentered,
    "radon_hierarchical_intercept_centered": RadonHierarchicalInterceptCentered,
    "radon_hierarchical_intercept_noncentered": RadonHierarchicalInterceptNoncentered,
    "seeds_model": SeedsModel,
    "seeds_centered_model": SeedsCenteredModel,
    "seeds_stanified_model": SeedsStanifiedModel,
    "surgical_model": SurgicalModel,
    "GLM_Poisson_model": GLMPoissonModel,
    "arma11": Arma11,
    "pilots": Pilots,
    "nes": Nes,
    "dogs": Dogs,
    "dogs_log": DogsLog,
    "irt_2pl": Irt2pl,
    "gp_regr": GpRegr,
    "rats_model": RatsModel,
    "GLMM1_model": GLMM1Model,
    "GLM_Binomial_model": GLMBinomialModel,
    "GLMM_Poisson_model": GLMMPoissonModel,
    "diamonds": Diamonds,
    "hier_2pl": Hier2pl,
    "M0_model": M0Model,
    "logistic_regression_rhs": LogisticRegressionRhs,
    "Mt_model": MtModel,
    "bym2_offset_only": Bym2OffsetOnly,
    "lsat_model": LsatModel,
}
