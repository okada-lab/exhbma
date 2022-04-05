import logging

import numpy as np

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

        Notes
        -----
        Training data X and y are copied so that original data are not modified.
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
        XTy = np.dot(X.T, y)

        mu = self._calculate_coefficient(X_svd_vh=vh, X_svd_s=s, XTy=XTy)
        self.coef_ = mu.tolist()

        self.log_likelihood_ = self._calculate_log_likelihood(X_svd_u=u, X_svd_s=s, y=y)

    def _calculate_coefficient(
        self, X_svd_s: np.ndarray, X_svd_vh: np.ndarray, XTy: np.ndarray
    ):
        """
        Calculate coefficient using SVD component of X.
        X = u @ np.diag(s) @ vh

        Lambda = vh.T (np.diag(s)**2 / sigma_noise**2 + 1/sigma_coef**2) vh
        mu = Lambda**(-1) X.T y / sigma_noise**2
           = vh.T (np.diag(s) ** 2 + sigma_noise**2 / sigma_coef ** 2)**(-1) vh X.T y
        """
        n_features = len(XTy)
        eigvals_XTX = np.zeros(n_features)
        eigvals_XTX[: len(X_svd_s)] = X_svd_s ** 2
        # Put sigma_noise into eigvals_lambda
        eigvals_lambda = eigvals_XTX + self.sigma_noise ** 2 / self.sigma_coef ** 2

        mu = np.dot(X_svd_vh.T, np.dot(X_svd_vh, XTy) / eigvals_lambda)
        return mu

    def _calculate_log_likelihood(
        self, X_svd_u: np.ndarray, X_svd_s: np.ndarray, y: np.ndarray
    ):
        """
        Calculate log likelihood using SVD component of X.
        X = u @ np.diag(s) @ vh

        log p = - N/2 log(2 pi)
                - 1/2 log det (sigma_noise**2 I + sigma_coef**2 X XT)
                - 1/2 yT (sigma_noise**2 I + sigma_coef**2 X XT)**(-1) y
                + 1/2 log(sigma_noise**2 / (N + sigma_noise**2))

        sigma_noise**2 I + sigma_coef**2 X XT
        = u (sigma_noise**2 I + sigma_coef**2 np.diag(s)**2) u.T
        """
        n_data = len(y)
        eigvals_XXT = np.zeros(n_data)
        eigvals_XXT[: len(X_svd_s)] = X_svd_s ** 2
        eigvals_cov = self.sigma_noise ** 2 + self.sigma_coef ** 2 * eigvals_XXT
        uTy = np.dot(X_svd_u.T, y)

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

    def _validate_training_data_shape(self, X, y):
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
        if self.coef_ is None:
            raise ValueError(
                "`fit` is not executed. Prediction needs `fit` in advance."
            )

        pred = np.dot(X, self.coef_)
        return pred
