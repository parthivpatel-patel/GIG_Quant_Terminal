"""Research models that are fitted in-process. None of these are champions."""

from gig.models.garch import ewma_variance, fit_garch11
from gig.models.hmm import fit_gaussian_hmm
from gig.models.kalman import kalman_dynamic_regression

__all__ = [
    "ewma_variance",
    "fit_garch11",
    "fit_gaussian_hmm",
    "kalman_dynamic_regression",
]
