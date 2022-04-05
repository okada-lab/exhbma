import pkg_resources

from .exhaustive_search import ExhaustiveLinearRegression
from .integrate import (
    integrate_log_values_in_line,
    integrate_log_values_in_square,
    validate_list_dimension,
)
from .linear_regression import LinearRegression
from .plot import feature_posterior, sigma_posterior, weight_diagram
from .probabilities import RandomVariable, gamma, uniform
from .scaler import StandardScaler

try:
    __version__ = pkg_resources.get_distribution("exhbma").version
except pkg_resources.DistributionNotFound:
    # Only for document generation
    __version__ = "dev"
