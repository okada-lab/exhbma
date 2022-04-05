import logging
from enum import Enum, auto
from itertools import product
from typing import List, Tuple

import numpy as np
from pydantic import BaseModel, Field
from scipy.special import logsumexp
from tqdm import tqdm

from exhbma.integrate import integrate_log_values_in_square
from exhbma.linear_regression import LinearRegression
from exhbma.probabilities import RandomVariable

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ModelInfo(BaseModel):
    indicator: List[int] = Field(
        ...,
        description="Indicator vector of the model. This attribute may be excluded in the future, please use parent's `indicators_` instead.",  # noqa
    )
    log_prior: float = Field(
        ...,
        description="Log-prior of the model. This attribute may be excluded in the future, please use parent's `log_priors_` instead.",  # noqa
    )
    coefficient: List[float] = Field(
        ...,
        description="Coefficient of linear model, which is marginalized over sigma_noise and sigma_coef.",  # noqa
    )
    log_likelihood: float = Field(
        ..., description="Marginal log-likelihood of the model."
    )
    log_likelihood_over_sigma: List[List[float]] = Field(
        ...,
        description="Log-likelihood over sigma_noise and sigma_coef, `p(y| sigma_noise, sigma_coef, X)`.",  # noqa
    )


class PredictMode(Enum):
    select = auto()
    full = auto()


class ExhaustiveLinearRegression(object):
    r"""
    ExhaustiveSearchModel with linear_model.LinearRegression

    - Intercept of the linear model is assumed to be zero.

        - Assume that target variable y is centralized.
        - Assume that all features x are centralized and normalized.

    Parameters
    ----------
    sigma_noise_points: List[RandomVariable]
        Data points to explore sigma_noise parameter in exhaustive search.

    sigma_coef_points: List[RandomVariable]
        Data points to explore sigma_coef parameter in exhaustive search.

    alpha: float (default: 0.5)
        Alpha parameter is fixed to this value.

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
        :math:`p(y| \sigma_{noise}, \sigma_{coef}, X)`,
        which is marginalized over indicators.
        Prior distributions for both sigma are not included.

    feature_posteriors_: List[float]
        Posterior probabilities for each feature.

    indicators_: List[List[int]]
        List of indicator vectors.
        Null model `[0, 0, ..., 0]` are excluded,
        so length is :math:`2^{n\_features\_in\_} - 1`.

    log_priors_: List[float]
        List of log-prior probabilities for each model specified by indicator.

    log_likelihoods_: List[float]
        List of log-likelihood of each model specified by indicator.

    models_: List[ModelInfo]
        Information for all models specified by indicator vector.
        Length of this attribute is equal to that of indicators_
        and models correspond to each other.

    Notes
    -----
    Hello World!
    """

    def __init__(
        self,
        sigma_noise_points: List[RandomVariable],
        sigma_coef_points: List[RandomVariable],
        alpha: float = 0.5,
    ):
        self.sigma_noise_points = sigma_noise_points
        self.sigma_coef_points = sigma_coef_points
        self.alpha = alpha
        self._preprocessing_tolerance = 1e-8

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        """Train a model

        1. Create indicator vectors
        2. Fit sub-models for each indicator vector
           with marginalizing sigma_noise and sigma_coef for each indicator
        3. Calculate final model averaged over sub-models

        Parameters
        ----------
        X : np.ndarray with shape (n_data, n_features)
            Feature matrix. Each row corresponds to single data.

        y : np.ndarray with shape  (n_data,)
            Target value vector.

        verbose: bool (default: True)
            If this is set to `True`, progress bar is displayed.
        """
        self.n_features_in_: int = X.shape[1]

        LinearRegression.validate_target_centralization(
            y=y, tolerance=self._preprocessing_tolerance
        )
        LinearRegression.validate_feature_standardization(
            X=X, tolerance=self._preprocessing_tolerance
        )

        self.indicators_: List[List[int]] = self._generate_indicator(
            n_features=self.n_features_in_, exclude_null=True
        )
        self.log_priors_: List[float] = []
        self.log_likelihoods_: List[float] = []
        self.models_: List[ModelInfo] = []
        for indicator in tqdm(self.indicators_, disable=not verbose):
            (
                log_likelihood,
                log_likelihood_over_sigma,
                coefficient,
            ) = self._fit_over_sigma_noise_and_coef(
                X=X[:, np.array(indicator) == 1], y=y
            )
            self.log_priors_.append(self._fixed_alpha_prior(indicator=indicator))
            self.log_likelihoods_.append(log_likelihood)
            self.models_.append(
                ModelInfo(
                    indicator=indicator,
                    log_prior=self.log_priors_[-1],
                    coefficient=coefficient,
                    log_likelihood_over_sigma=log_likelihood_over_sigma,
                    log_likelihood=log_likelihood,
                )
            )

        self.log_likelihood_: float = self._calculate_log_marginal_likelihood(
            log_priors=self.log_priors_, models=self.models_
        )
        self.feature_posteriors_: List[float] = self._calculate_feature_posterior(
            log_priors=self.log_priors_,
            indicators=self.indicators_,
            models=self.models_,
        )
        self.log_likelihood_over_sigma_: List[
            List[float]
        ] = self._calculate_log_marginal_likelihood_over_sigma(
            log_priors=self.log_priors_, models=self.models_
        )

        coefficient = self._calculate_marginal_linear_model(
            log_priors=self.log_priors_,
            indicators=self.indicators_,
            models=self.models_,
        )
        self.coef_: List[float] = coefficient

    def _fit_over_sigma_noise_and_coef(
        self, X, y
    ) -> Tuple[float, List[List[float]], List[float]]:
        """
        Fit over (sigma_noise, sigma_coef) grid points and
        calculate log model likelihood by marginalizing.
        """
        fit_models: List[List[LinearRegression]] = []
        for sigma_noise in self.sigma_noise_points:
            models_along_coef = []
            for sigma_coef in self.sigma_coef_points:
                reg = LinearRegression(
                    sigma_noise=sigma_noise.position,
                    sigma_coef=sigma_coef.position,
                )
                reg.fit(X, y, skip_preprocessing_validation=True)
                models_along_coef.append(reg)
            fit_models.append(models_along_coef)

        # Prepare for marginalization
        np_log_likelihood_over_sigma = np.array(
            [
                [model.log_likelihood_ for model in models_along_coef]
                for models_along_coef in fit_models
            ]
        )
        np_log_prior = np.log([p.prob for p in self.sigma_noise_points]).reshape(
            -1, 1
        ) + np.log([p.prob for p in self.sigma_coef_points]).reshape(1, -1)
        each_model_log_joint_probabilities = np_log_likelihood_over_sigma + np_log_prior

        # Marginalize over sigma_noise and sigma_coef
        log_likelihood = integrate_log_values_in_square(
            log_values=each_model_log_joint_probabilities.tolist(),
            x1=[p.position for p in self.sigma_noise_points],
            x2=[p.position for p in self.sigma_coef_points],
        )

        # Integrate coefficient by log_prob_values
        # More specifically, calculated coefficient is the mean of p(w|c, X, y).
        coefficient = []
        for i in range(X.shape[1]):
            coefficient_weights = [
                [model.coef_[i] for model in models_along_coef]
                for models_along_coef in fit_models
            ]
            result = integrate_log_values_in_square(
                log_values=(
                    each_model_log_joint_probabilities - log_likelihood
                ).tolist(),
                x1=[p.position for p in self.sigma_noise_points],
                x2=[p.position for p in self.sigma_coef_points],
                weights=coefficient_weights,
                expect_positive=False,
            )
            coefficient.append(result[1] * np.exp(result[0]))

        return (
            log_likelihood,
            np_log_likelihood_over_sigma.tolist(),
            coefficient,
        )

    def _calculate_log_marginal_likelihood(
        self, log_priors: List[float], models: List[ModelInfo]
    ) -> float:
        log_likelihood = logsumexp(
            [p + m.log_likelihood for (p, m) in zip(log_priors, models)]
        )
        return log_likelihood

    def _calculate_feature_posterior(
        self,
        log_priors: List[float],
        indicators: List[List[int]],
        models: List[ModelInfo],
    ) -> List[float]:
        log_joint_probabilities = [
            p + m.log_likelihood for (p, m) in zip(log_priors, models)
        ]
        log_marginal_likelihood = self._calculate_log_marginal_likelihood(
            log_priors=log_priors, models=models
        )
        np_indicators = np.array(indicators)
        log_marginals = []
        for column in range(np_indicators.shape[1]):
            indicator_values = np_indicators[:, column]
            log_marginals.append(
                logsumexp(a=log_joint_probabilities, b=indicator_values)
                - log_marginal_likelihood
            )
        return np.exp(log_marginals).tolist()

    def _calculate_log_marginal_likelihood_over_sigma(
        self, log_priors: List[float], models: List[ModelInfo]
    ) -> List[List[float]]:
        log_likelihood_over_sigma = logsumexp(
            [
                np.array(m.log_likelihood_over_sigma) + p
                for (m, p) in zip(models, log_priors)
            ],
            axis=0,
        )
        return log_likelihood_over_sigma.tolist()

    def _calculate_marginal_linear_model(
        self,
        log_priors: List[float],
        indicators: List[List[int]],
        models: List[ModelInfo],
    ) -> List[float]:
        log_joint_probabilities = [
            p + m.log_likelihood for (p, m) in zip(log_priors, models)
        ]
        log_marginal_likelihood = self._calculate_log_marginal_likelihood(
            log_priors=log_priors, models=models
        )
        np_coefficient = np.zeros((len(models), self.n_features_in_))
        for i, (m, indicator) in enumerate(zip(models, indicators)):
            np_coefficient[i, np.array(indicator) == 1] = m.coefficient

        coefficient = []
        for column in range(np_coefficient.shape[1]):
            coefficient_values = np_coefficient[:, column]
            result = logsumexp(
                a=log_joint_probabilities, b=coefficient_values, return_sign=True
            )
            coefficient.append(result[1] * np.exp(result[0] - log_marginal_likelihood))

        return coefficient

    def _fixed_alpha_prior(self, indicator: List[int]) -> float:
        """
        Model prior with fixed alpha:
        p(c) = prod_{i=1}^p alpha^{c_i} (1-alpha)^{1-c_i}
        """
        n_features = len(indicator)
        n_in_use = sum(indicator)
        log_model_prior = n_in_use * np.log(self.alpha) + (
            n_features - n_in_use
        ) * np.log(1 - self.alpha)
        return log_model_prior

    def _generate_indicator(
        self, n_features: int, exclude_null: bool = True
    ) -> List[List[int]]:
        """
        Parameters
        ----------
        n_features : int
            Number of features.

        exclude_null: bool
            Whether to exclude null model ([0, 0, .., 0]).

        Returns
        -------
        indicators : List with shape (n_combinations, n_features)
            All combinations of indicator vector.
            n_combinations is 2**n_features - 1 if exclude_null = True,
            else 2**n_features.
        """
        start = 1 if exclude_null else 0
        indicators = [list(p)[::-1] for p in product([0, 1], repeat=n_features)]
        return indicators[start:]

    def _transform_indicator_to_model_index(self, indicator: List[int]) -> int:
        if len(indicator) != self.n_features_in_:
            raise ValueError(
                f"indicator should have be {self.n_features_in_}-length list"
            )
        bin_expr = "".join(map(str, indicator))[::-1]
        return int(bin_expr, 2) - 1

    def predict(self, X: np.ndarray, mode: str, threshold: float = 0.5) -> np.ndarray:
        """Predict using the model

        Predict about new data.

        Parameters
        ----------
        X : np.ndarray with shape (n_data, n_features)
            Feature matrix for prediction.

        mode: str
            Mode of prediction.

        threshold: float (default: 0.5)
            Feature selection threshold used in mode 'select'.

        Returns
        -------
        y: np.ndarray
            Prediction values.
        """
        try:
            predict_mode = PredictMode[mode]
        except KeyError:
            raise ValueError(
                "Invalid mode: `{}` specified. Mode should be either of {}".format(
                    mode, list(PredictMode.__members__)
                )
            )

        if predict_mode == PredictMode.select:
            pred = self._predict_by_select(X=X, threshold=threshold)
        elif predict_mode == PredictMode.full:
            pred = self._predict_by_full(X=X)
        return pred

    def _predict_by_select(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        indicator = self.select_variables(threshold=threshold)
        model = self.models_[
            self._transform_indicator_to_model_index(indicator=indicator)
        ]
        pred = np.dot(X[:, np.array(indicator) == 1], model.coefficient)
        return pred

    def _predict_by_full(self, X: np.ndarray) -> np.ndarray:
        pred = np.dot(X, self.coef_)
        return pred

    def select_variables(self, threshold: float = 0.5) -> List[int]:
        """
        Return indicator with posterior probability greater than or equal to threshold.
        """
        index = np.array(self.feature_posteriors_) >= threshold
        return index.astype(int).tolist()
