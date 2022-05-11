import logging
from typing import List

import numpy as np

from exhbma.integrate import integrate_log_values_in_square
from exhbma.probabilities import RandomVariable

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LinearRegression(object):
    """
    Model description:

    - Base model: Linear regression model
    - Observation noise: Gaussian distribution
    - Prior distribution for coefficient: Gaussian distribution
    - Intercept of the linear model is assumed to be zero.

        - Assume that target variable y is centralized.
        - Assume that all features x are centralized and normalized.
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
        Number of features seen during fit.

    coef_: List[float]
        Coefficients of the regression model (mean of distribution).

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
        Calculate coefficient used in prediction and log likelihood for this data.

        Parameters
        ----------
        X : np.ndarray with shape (n_data, n_features)
            Feature matrix. Each row corresponds to single data.

        y : np.ndarray with shape  (n_data,)
            Target value vector.
        """
        self._validate_training_data_shape(X, y)
        if not skip_preprocessing_validation:
            self.validate_target_centralization(
                y, tolerance=self._preprocessing_tolerance
            )
            self.validate_feature_standardization(
                X, tolerance=self._preprocessing_tolerance
            )
        self.n_features_in_ = X.shape[1]

        u, s, vh = np.linalg.svd(X, full_matrices=True)
        uTy = np.dot(u.T, y)
        XTy = np.dot(X.T, y)
        vhXTy = np.dot(vh, XTy)

        self._fit_after_svd(uTy=uTy, s=s, vh=vh, vhXTy=vhXTy, y=y)

    def _fit_after_svd(
        self,
        uTy: np.ndarray,
        s: np.ndarray,
        vh: np.ndarray,
        vhXTy: np.ndarray,
        y: np.ndarray,
    ):
        """
        Calculate coefficient and log-likelihood using SVD component of X.
        """
        mu = self._calculate_coefficient(vh=vh, s=s, vhXTy=vhXTy)
        self.coef_ = mu.tolist()

        self.log_likelihood_ = self._calculate_log_likelihood(uTy=uTy, s=s, y=y)

    def _calculate_coefficient(self, s: np.ndarray, vh: np.ndarray, vhXTy: np.ndarray):
        """
        Calculate coefficient using SVD component of X.
        X = u @ np.diag(s) @ vh

        Lambda = vh.T (np.diag(s)**2 / sigma_noise**2 + 1/sigma_coef**2) vh
        mu = Lambda**(-1) X.T y / sigma_noise**2
           = vh.T (np.diag(s) ** 2 + sigma_noise**2 / sigma_coef ** 2)**(-1) vh X.T y
        """
        n_features = vh.shape[1]
        eigvals_XTX = np.zeros(n_features)
        eigvals_XTX[: len(s)] = s ** 2
        # Put sigma_noise into eigvals_lambda
        eigvals_lambda = eigvals_XTX + self.sigma_noise ** 2 / self.sigma_coef ** 2

        mu = np.dot(vh.T, vhXTy / eigvals_lambda)
        return mu

    def _calculate_log_likelihood(self, uTy: np.ndarray, s: np.ndarray, y: np.ndarray):
        """
        Calculate log likelihood using SVD component of X.
        X = u @ np.diag(s) @ vh

        log p = - N/2 log(2 pi)
                - 1/2 log det (sigma_noise**2 I + sigma_coef**2 X XT)
                - 1/2 yT (sigma_noise**2 I + sigma_coef**2 X XT)**(-1) y
                + 1/2 log(sigma_noise**2 / (N * var(y) + sigma_noise**2))

        sigma_noise**2 I + sigma_coef**2 X XT
        = u (sigma_noise**2 I + sigma_coef**2 np.diag(s)**2) u.T
        """
        n_data = len(y)
        eigvals_XXT = np.zeros(n_data)
        eigvals_XXT[: len(s)] = s ** 2
        eigvals_cov = self.sigma_noise ** 2 + self.sigma_coef ** 2 * eigvals_XXT

        const = -n_data / 2 * np.log(2 * np.pi)
        log_det = -1 / 2 * np.log(eigvals_cov).sum()
        log_exp = -1 / 2 * np.dot(uTy, uTy / eigvals_cov)
        var_y = np.var(y)
        log_intercept = (
            np.log(self.sigma_noise ** 2)
            - np.log(n_data * var_y + self.sigma_noise ** 2)
        ) / 2
        log_likelihood = const + log_det + log_exp + log_intercept
        return log_likelihood

    @staticmethod
    def _validate_training_data_shape(X, y):
        """
        Validate training data X and y.
        Check list:
        - X is 2 dimension array.
        - X and y have same number of data.
        """
        if len(X.shape) != 2:
            raise ValueError(
                "X is expect to be 2-dim array. Actual {}-dim.".format(len(X.shape))
            )
        if len(X) != len(y):
            raise ValueError(
                "Data sizes are different between X({}) and y({}).".format(
                    len(X), len(y)
                )
            )

    @staticmethod
    def validate_feature_standardization(X, tolerance: float = 1e-8):
        for i in range(X.shape[1]):
            x = X[:, i]
            x_mean = np.mean(x)
            x_var = np.var(x)
            if np.abs(x_mean) > tolerance:
                logger.error(
                    "Mean of %s-th feature(%s) is out of tolerance(%s)",
                    i,
                    x_mean,
                    tolerance,
                )
                raise ValueError(f"Feature in column-{i} is not centralized.")
            if np.abs(1 - x_var) > tolerance:
                logger.error(
                    "Variance of %s-th feature(%s) is out of tolerance(%s)",
                    i,
                    x_var,
                    tolerance,
                )
                raise ValueError(f"Feature in column-{i} is not normalized.")

    @staticmethod
    def validate_target_centralization(y, tolerance: float = 1e-8):
        y_mean = np.mean(y)
        if np.abs(y_mean) > tolerance:
            logger.error(
                "Mean of target variable(%s) is out of tolerance(%s)",
                y_mean,
                tolerance,
            )
            raise ValueError("Target variable is not centralized.")

    def predict(self, X):
        """
        Prediction using trained model.

        X : np.ndarray with shape (n_data, n_features)
            Feature matrix for prediction.
        """
        pred = np.dot(X, self.coef_)
        return pred


class ConstantRegression(object):
    """
    Model description:

    - Base model: Constant model
    - Observation noise: Gaussian distribution
    - Prior distribution for coefficient: Gaussian distribution
    - Intercept of the linear model is assumed to be zero.

        - Assume that target variable y is centralized.
        - Assume that all features x are centralized and normalized.
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
        Number of features seen during fit.

    coef_: List[float]
        Coefficients of the regression model (mean of distribution).

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
        X : np.ndarray with shape (n_data, n_features)
            Feature matrix. Each row corresponds to single data.

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
        return 0


class MarginalLinearRegression(object):
    r"""
    Model description:

    - Base model: Linear regression model
    - Observation noise: Gaussian distribution
    - Prior distribution for coefficient: Gaussian distribution
    - Intercept of the linear model is assumed to be zero.
    - Marginalization over sigma_noise and sigma_coefficient are performed.

        - Assume that target variable y is centralized.
        - Assume that all features x are centralized and normalized.
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
        Number of features seen during fit.

    coef_: List[float]
        Coefficients of the regression model (mean of distribution).

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
        Calculate coefficient used in prediction and log likelihood for this data.

        Parameters
        ----------
        X : np.ndarray with shape (n_data, n_features)
            Feature matrix. Each row corresponds to single data.

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
        self.n_features_in_ = X.shape[1]

        # Fit models
        fit_models: List[List[LinearRegression]] = self._fit_models_over_sigma(X=X, y=y)

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

        # Integrate coefficient by log_prob_values
        coefficient = []
        for i in range(X.shape[1]):
            coefficient.append(
                self._integrate_coefficient_over_sigma(
                    index=i,
                    fit_models=fit_models,
                    log_joint_probabilities=log_joint_probabilities,
                    log_likelihood=log_likelihood,
                )
            )

        self.log_likelihood_ = log_likelihood
        self.log_likelihood_over_sigma_ = log_likelihood_over_sigma.tolist()
        self.coef_ = coefficient

    def _fit_models_over_sigma(self, X, y) -> List[List[LinearRegression]]:
        u, s, vh = np.linalg.svd(X, full_matrices=True)
        uTy = np.dot(u.T, y)
        XTy = np.dot(X.T, y)
        vhXTy = np.dot(vh, XTy)

        fit_models: List[List[LinearRegression]] = []
        for sigma_noise in self.sigma_noise_points:
            models_along_coef = []
            for sigma_coef in self.sigma_coef_points:
                reg = LinearRegression(
                    sigma_noise=sigma_noise.position, sigma_coef=sigma_coef.position
                )
                reg._fit_after_svd(uTy=uTy, s=s, vh=vh, vhXTy=vhXTy, y=y)
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

    def _integrate_coefficient_over_sigma(
        self,
        index: int,
        fit_models: List[List[LinearRegression]],
        log_joint_probabilities,
        log_likelihood: float,
    ) -> float:
        coefficient_weights = [
            [model.coef_[index] for model in models_along_coef]
            for models_along_coef in fit_models
        ]
        result = integrate_log_values_in_square(
            log_values=(log_joint_probabilities - log_likelihood).tolist(),
            x1=[p.position for p in self.sigma_noise_points],
            x2=[p.position for p in self.sigma_coef_points],
            weights=coefficient_weights,
            expect_positive=False,
        )
        return result[1] * np.exp(result[0])

    def predict(self, X):
        """
        Prediction using trained model.

        X : np.ndarray with shape (n_data, n_features)
            Feature matrix for prediction.
        """
        pred = np.dot(X, self.coef_)
        return pred


class MarginalNullRegression(object):
    r"""
    Model description:

    - Base model: Constant model
    - Observation noise: Gaussian distribution
    - Intercept of the linear model is assumed to be zero.
    - Marginalization over sigma_noise and sigma_coefficient are performed.

        - Assume that target variable y is centralized.
        - Assume that all features x are centralized and normalized.
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
        Number of features seen during fit.

    coef_: List[float]
        Coefficients of the regression model (mean of distribution).

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
        self.n_features_in_ = X.shape[1]

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
        self.coef_: List[float] = []

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
        return 0
