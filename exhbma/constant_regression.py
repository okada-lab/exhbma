import logging
from typing import List

import numpy as np

from exhbma.integrate import integrate_log_values_in_square
from exhbma.linear_regression import LinearRegression
from exhbma.probabilities import RandomVariable

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ConstantRegression(object):
    """
    Model description:

    - Base model: Constant model
    - Observation noise: Gaussian distribution
    - Intercept of the model is assumed to be zero.

        - Assume that target variable y is centralized.
        - Marginalization over intercept is performed.

    Parameters
    ----------
    sigma_noise: float
        Standard deviation of gaussian noise.
        Model assumes that observation noise obeys Gaussian distribution
        with mean = 0, variance = sigma_noise^2.

    sigma_coef: float
        Standard deviation of gaussian noise.
        Model assumes that each coefficient value obeys Gaussian distribution
        with mean = 0, variance = sigma_coef^2 independently.

    Attributes
    ----------
    n_features_in_: int
        Always return 0 value, but prepared to unify api for regression models.

    coef_: List[float]
        Always return null list [], but prepared to unify api for regression models.

    log_likelihood_: float
        Log-likelihood of the model.
        Marginalization is performed over sigma_noise, sigma_coef, indicators.
    """

    def __init__(self, sigma_noise: float, sigma_coef: float):
        self.sigma_noise = sigma_noise
        self.sigma_coef = sigma_coef
        self._preprocessing_tolerance = 1e-8

    def fit(
        self, X: np.ndarray, y: np.ndarray, skip_preprocessing_validation: bool = False
    ):
        """
        Calculate log likelihood for this data.

        Parameters
        ----------
        X : np.ndarray with shape (n_data, 0)
            Null feature matrix.

        y : np.ndarray with shape  (n_data,)
            Target value vector.
        """
        if not skip_preprocessing_validation:
            LinearRegression.validate_target_centralization(
                y, tolerance=self._preprocessing_tolerance
            )
            LinearRegression.validate_feature_standardization(
                X, tolerance=self._preprocessing_tolerance
            )
        self.n_features_in_ = 0

        self.coef_: List[float] = []
        self.log_likelihood_ = self._calculate_log_likelihood(y=y)

    def _calculate_log_likelihood(self, y: np.ndarray):
        """
        Calculate log Likelihood for constant model.
        log p = - N/2 log(2 pi * sigma_noise**2)
                - 1/2 yTy**2 / sigma_noise**2
                + 1/2 log(sigma_noise**2 / (N var(y) + sigma_noise**2))
        """
        n_data = len(y)

        const = -n_data / 2 * (np.log(2 * np.pi) + np.log(self.sigma_noise ** 2))
        log_exp = -1 / 2 * np.dot(y, y) / self.sigma_noise ** 2
        var_y = np.var(y)
        log_intercept = (
            np.log(self.sigma_noise ** 2)
            - np.log(n_data * var_y + self.sigma_noise ** 2)
        ) / 2
        log_likelihood = const + log_exp + log_intercept
        return log_likelihood

    def predict(self, X):
        """
        Since data is centralized, this method always returns 0 value.
        """
        return np.zeros(X.shape[0])


class MarginalConstantRegression(object):
    r"""
    Model description:

    - Base model: Constant model
    - Observation noise: Gaussian distribution
    - Intercept of the model is assumed to be zero.
    - Marginalization over sigma_noise and sigma_coefficient are performed.

        - Assume that target variable y is centralized.
        - Marginalization over intercept is performed.

    Parameters
    ----------
    sigma_noise_points: List[RandomVariable]
        Data points to marginalize over sigma_noise parameter.

    sigma_coef_points: List[RandomVariable]
        Data points to marginalize over sigma_coef parameter.


    Attributes
    ----------
    n_features_in_: int
        Always return 0 value, but prepared to unify api for regression models.

    coef_: List[float]
        Always return null list [], but prepared to unify api for regression models.

    log_likelihood_: float
        Log-likelihood of the model.
        Marginalization is performed over sigma_noise, sigma_coef, indicators.

    log_likelihood_over_sigma_: List[List[float]]
        Log-likelihood over :math:`\sigma_{noise}` and :math:`\sigma_{coef}`,
        :math:`p(y| \sigma_{noise}, \sigma_{coef}, X)`.
        Prior distributions for both sigma are not included.
    """

    def __init__(
        self,
        sigma_noise_points: List[RandomVariable],
        sigma_coef_points: List[RandomVariable],
    ):
        self.sigma_noise_points = sigma_noise_points
        self.sigma_coef_points = sigma_coef_points
        self._preprocessing_tolerance = 1e-8

    def fit(
        self, X: np.ndarray, y: np.ndarray, skip_preprocessing_validation: bool = False
    ):
        """
        Calculate log likelihood for this data.

        Parameters
        ----------
        X : np.ndarray with shape (n_data, 0)
            Null feature matrix.

        y : np.ndarray with shape  (n_data,)
            Target value vector.
        """
        LinearRegression._validate_training_data_shape(X, y)
        if not skip_preprocessing_validation:
            LinearRegression.validate_target_centralization(
                y, tolerance=self._preprocessing_tolerance
            )
            LinearRegression.validate_feature_standardization(
                X, tolerance=self._preprocessing_tolerance
            )
        self.n_features_in_ = 0
        self.coef_: List[float] = []

        # Fit models
        fit_models: List[List[ConstantRegression]] = self._fit_models_over_sigma(
            X=X, y=y
        )

        # Calculate log likelihood
        log_likelihood_over_sigma = np.array(
            [
                [model.log_likelihood_ for model in models_along_coef]
                for models_along_coef in fit_models
            ]
        )
        log_prior = np.log([p.prob for p in self.sigma_noise_points]).reshape(
            -1, 1
        ) + np.log([p.prob for p in self.sigma_coef_points]).reshape(1, -1)
        log_joint_probabilities = log_likelihood_over_sigma + log_prior

        log_likelihood = self._integrate_log_likelihood_over_sigma(
            log_joint_probabilities=log_joint_probabilities
        )

        self.log_likelihood_ = log_likelihood
        self.log_likelihood_over_sigma_ = log_likelihood_over_sigma.tolist()

    def _fit_models_over_sigma(self, X, y) -> List[List[ConstantRegression]]:
        fit_models: List[List[ConstantRegression]] = []
        for sigma_noise in self.sigma_noise_points:
            models_along_coef = []
            for sigma_coef in self.sigma_coef_points:
                reg = ConstantRegression(
                    sigma_noise=sigma_noise.position, sigma_coef=sigma_coef.position
                )
                reg.fit(X=X, y=y, skip_preprocessing_validation=True)
                models_along_coef.append(reg)
            fit_models.append(models_along_coef)
        return fit_models

    def _integrate_log_likelihood_over_sigma(
        self,
        log_joint_probabilities,
    ) -> float:
        # Marginalize over sigma_noise and sigma_coef
        log_likelihood = integrate_log_values_in_square(
            log_values=log_joint_probabilities.tolist(),
            x1=[p.position for p in self.sigma_noise_points],
            x2=[p.position for p in self.sigma_coef_points],
        )
        return log_likelihood

    def predict(self, X):
        """
        Since data is centralized, this method always returns 0 value.
        """
        return np.zeros(X.shape[0])
