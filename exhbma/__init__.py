__version__ = "0.1.0"

from .exhaustive_search import ExhaustiveLinearRegression
from .integrate import (
    integrate_log_values_in_line,
    integrate_log_values_in_square,
    validate_list_dimension,
)
from .linear_regression import LinearRegression
from .plot import feature_posterior, sigma_posterior, weight_diagram
from .probabilities import BetaDistributionParams, RandomVariable, gamma, uniform
from .scaler import StandardScaler
