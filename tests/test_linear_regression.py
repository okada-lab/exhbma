from typing import Union

import numpy as np
import pytest
from scipy.special import logsumexp

from exhbma import (
    ConstantRegression,
    LinearRegression,
    MarginalLinearRegression,
    RandomVariable,
    StandardScaler,
)


@pytest.fixture()
def seed():
    val = 0
    np.random.seed(val)
    return val


def check_linear_model(
    model: Union[LinearRegression, MarginalLinearRegression], expect_coef: np.ndarray
):
    assert model.coef_ == pytest.approx(expect_coef, rel=1e-1)


def calculate_rmse(true_y, pred_y) -> float:
    return np.power(true_y - pred_y, 2).mean() ** 0.5


def _calculate_analytical_form(X, y, sigma_noise: float, sigma_coef: float):
    """Calculation by analytical form"""
    n_data = X.shape[0]

    lambda_matrix = (
        np.dot(X.T, X) / sigma_noise ** 2 + np.eye(X.shape[1]) / sigma_coef ** 2
    )
    mu = np.dot(np.linalg.inv(lambda_matrix), np.dot(X.T, y) / sigma_noise ** 2)

    cov_matrix = sigma_noise ** 2 * np.eye(n_data) + sigma_coef ** 2 * np.dot(X, X.T)
    var_y = np.var(y)
    log_likelihood = (
        -n_data / 2 * np.log(2 * np.pi)
        - 1 / 2 * np.log(np.linalg.det(cov_matrix))
        - 1 / 2 * np.dot(y, np.dot(np.linalg.inv(cov_matrix), y))
        + 1 / 2 * np.log(sigma_noise ** 2 / (n_data * var_y + sigma_noise ** 2))
    )
    return mu, log_likelihood


def compare_straightforward_analytical_form(
    model: LinearRegression,
    X,
    y,
    sigma_noise: float,
    sigma_coef: float,
):
    """
    Compare against straightforward analytical form because in LinearRegression class,
    calculation are performed more effectively.
    """
    mu, log_likelihood = _calculate_analytical_form(
        X=X,
        y=y,
        sigma_noise=sigma_noise,
        sigma_coef=sigma_coef,
    )

    assert model.coef_ == pytest.approx(mu)
    assert model.log_likelihood_ == pytest.approx(log_likelihood)


def test_linear_model_case_large_n_data(seed):
    """
    Test method `fit` by comparing against true linear model and test data.
    Noises are added so assertAlmostEqual's `places` is set loose.
    """
    n_data, n_features = 200, 10
    n_test = 1000
    sigma_noise = 0.01
    sigma_coef = 1

    X = np.random.randn(n_data, n_features)
    w = np.random.randn(n_features)
    y = np.dot(X, w) + np.random.randn(n_data) * sigma_noise

    test_X = np.random.randn(n_test, n_features)
    test_y = np.dot(test_X, w)

    x_scaler = StandardScaler(n_dim=2)
    y_scaler = StandardScaler(n_dim=1, scaling=False)
    x_scaler.fit(X)
    y_scaler.fit(y)
    X = x_scaler.transform(X)
    y = y_scaler.transform(y)

    reg = LinearRegression(sigma_noise=sigma_noise, sigma_coef=sigma_coef)
    reg.fit(X, y)

    assert reg.n_features_in_ == n_features

    check_linear_model(model=reg, expect_coef=w)

    pred_y = y_scaler.restore(reg.predict(x_scaler.transform(test_X)))
    assert calculate_rmse(test_y, pred_y) <= sigma_noise


def test_analytical_form_case_n_data_gt_n_features(seed):
    """
    Test method `fit` against straightforward analytical calculation.
    Situation: n_data > n_features (data rich situation)
    """
    n_data, n_features = 50, 20
    sigma_noise = 0.1
    sigma_coef = 1

    X = np.random.randn(n_data, n_features)
    w = np.random.randn(n_features)
    y = np.dot(X, w) + np.random.randn(n_data) * sigma_noise

    x_scaler = StandardScaler(n_dim=2)
    y_scaler = StandardScaler(n_dim=1, scaling=False)
    x_scaler.fit(X)
    y_scaler.fit(y)
    X = x_scaler.transform(X)
    y = y_scaler.transform(y)

    reg = LinearRegression(sigma_noise=sigma_noise, sigma_coef=sigma_coef)
    reg.fit(X, y)

    compare_straightforward_analytical_form(
        model=reg,
        X=X,
        y=y,
        sigma_noise=sigma_noise,
        sigma_coef=sigma_coef,
    )


def test_analytical_form_case_n_data_lt_n_features(seed):
    """
    Test method `fit` against straightforward analytical calculation.
    Situation: n_data < n_features (data needed situation)
    """
    n_data, n_features = 20, 50
    sigma_noise = 0.1
    sigma_coef = 1

    X = np.random.randn(n_data, n_features)
    w = np.random.randn(n_features)
    y = np.dot(X, w) + np.random.randn(n_data) * sigma_noise

    x_scaler = StandardScaler(n_dim=2)
    y_scaler = StandardScaler(n_dim=1, scaling=False)
    x_scaler.fit(X)
    y_scaler.fit(y)
    X = x_scaler.transform(X)
    y = y_scaler.transform(y)

    reg = LinearRegression(sigma_noise=sigma_noise, sigma_coef=sigma_coef)
    reg.fit(X, y)

    compare_straightforward_analytical_form(
        model=reg,
        X=X,
        y=y,
        sigma_noise=sigma_noise,
        sigma_coef=sigma_coef,
    )


def test_analytical_form_case_n_data_eq_n_features():
    """
    Test method `fit` against straightforward calculation.
    Situation: n_data == n_features
    """
    n_data, n_features = 30, 30
    sigma_noise = 0.1
    sigma_coef = 1
    np.random.seed(0)

    X = np.random.randn(n_data, n_features)
    w = np.random.randn(n_features)
    y = np.dot(X, w) + np.random.randn(n_data) * sigma_noise

    x_scaler = StandardScaler(n_dim=2)
    y_scaler = StandardScaler(n_dim=1, scaling=False)
    x_scaler.fit(X)
    y_scaler.fit(y)
    X = x_scaler.transform(X)
    y = y_scaler.transform(y)

    reg = LinearRegression(sigma_noise=sigma_noise, sigma_coef=sigma_coef)
    reg.fit(X, y)

    compare_straightforward_analytical_form(
        model=reg,
        X=X,
        y=y,
        sigma_noise=sigma_noise,
        sigma_coef=sigma_coef,
    )


def test_analytical_form_constant_regression(seed):
    """
    Test method `fit` of constant regression against
    straightforward analytical calculation.
    Situation: n_data > n_features (data rich situation)
    """
    n_data, n_features = 50, 20
    sigma_noise = 0.1
    sigma_coef = 1

    X = np.random.randn(n_data, n_features)
    w = np.random.randn(n_features)
    y = np.dot(X, w) + np.random.randn(n_data) * sigma_noise

    x_scaler = StandardScaler(n_dim=2)
    y_scaler = StandardScaler(n_dim=1, scaling=False)
    x_scaler.fit(X)
    y_scaler.fit(y)
    X = x_scaler.transform(X)
    y = y_scaler.transform(y)

    reg = ConstantRegression(sigma_noise=sigma_noise, sigma_coef=sigma_coef)
    reg.fit(X, y)

    log_likelihood = (
        -n_data / 2 * np.log(2 * np.pi * sigma_noise ** 2)
        - 1 / 2 * np.dot(y, y) / sigma_noise ** 2
        + 1 / 2 * np.log(sigma_noise ** 2 / (n_data * np.var(y) + sigma_noise ** 2))
    )

    assert [] == reg.coef_
    assert log_likelihood == reg.log_likelihood_


def test_validate_target_centralization(seed):
    """
    Test method `validate_target_centralization`.
    """
    n_data = 100
    tolerance = 1e-8

    y = np.random.randn(n_data)
    y = y - y.mean()
    LinearRegression.validate_target_centralization(y=y, tolerance=tolerance)


def test_validate_feature_standardization(seed):
    """
    Test method `validate_feature_standardization`.
    """
    n_data, n_features = 100, 10
    tolerance = 1e-8

    X = np.random.randn(n_data, n_features)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    LinearRegression.validate_feature_standardization(X=X, tolerance=tolerance)


def test_error_validate_target_centralization(seed):
    """
    Test error case of method `validate_target_centralization`.
    """
    n_data = 100
    offset = 1
    tolerance = 1e-8

    y = np.random.randn(n_data)
    y = y - y.mean() + offset
    with pytest.raises(ValueError, match="Target variable is not centralized."):
        LinearRegression.validate_target_centralization(y=y, tolerance=tolerance)


def test_error_validate_feature_centralization(seed):
    """
    Test error case of method `validate_feature_standardization`
    when feature is not centralized.
    """
    n_data, n_features = 100, 10
    target_index = 2
    offset = 1
    tolerance = 1e-8

    X = np.random.randn(n_data, n_features)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    X[:, target_index] += offset
    with pytest.raises(
        ValueError, match=f"Feature in column-{target_index} is not centralized."
    ):
        LinearRegression.validate_feature_standardization(X=X, tolerance=tolerance)


def test_error_validate_feature_normalization(seed):
    """
    Test error case of method `validate_feature_standardization`
    when feature is not normalized.
    """
    n_data, n_features = 100, 10
    target_index = 2
    scale = 1.5
    tolerance = 1e-8

    X = np.random.randn(n_data, n_features)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    X[:, target_index] *= scale
    with pytest.raises(
        ValueError, match=f"Feature in column-{target_index} is not normalized."
    ):
        LinearRegression.validate_feature_standardization(X=X, tolerance=tolerance)


def test_validate_training_data_shape():
    """
    Test method `_validate_training_data_shape`.
    """
    X = np.array([[1, 2], [2, 3], [3, 4]])
    y = np.array([3, 2, 1])

    LinearRegression._validate_training_data_shape(X, y)


def test_error_validate_X_shape_1dim():
    """
    Test error case of method `_validate_training_data_shape`
    when X's shape is 1 dimension.
    """
    X = np.array([1, 2, 3])
    y = np.array([3, 2, 1])

    reg = LinearRegression(sigma_noise=1, sigma_coef=1)

    with pytest.raises(
        ValueError, match="X is expect to be 2-dim array. Actual 1-dim."
    ):
        reg._validate_training_data_shape(X, y)


def test_error_validate_different_data_size():
    """
    Test error case of method `_validate_training_data_shape`
    when data size is different.
    """
    X = np.array([[1, 2], [2, 3], [3, 4]])
    y = np.array([3, 2, 1, 0])

    reg = LinearRegression(sigma_noise=1, sigma_coef=1)

    with pytest.raises(
        ValueError, match=r"Data sizes are different between X\(3\) and y\(4\)."
    ):
        reg._validate_training_data_shape(X, y)


def test_marginal_linear_regression(seed):
    """
    Test method `fit` in a simple case.
    """
    n_data, n_features = 200, 10
    n_test = 1000
    sigma_noise = 0.01

    X = np.random.randn(n_data, n_features)
    w = np.random.randn(n_features)
    y = np.dot(X, w) + np.random.randn(n_data) * sigma_noise

    test_X = np.random.randn(n_test, n_features)
    test_y = np.dot(test_X, w)

    x_scaler = StandardScaler(n_dim=2)
    y_scaler = StandardScaler(n_dim=1, scaling=False)
    x_scaler.fit(X)
    y_scaler.fit(y)
    X = x_scaler.transform(X)
    y = y_scaler.transform(y)

    sigma_noise_points = [
        RandomVariable(position=0.09, prob=50),
        RandomVariable(position=0.11, prob=50),
    ]
    sigma_coef_points = [
        RandomVariable(position=0.9, prob=5),
        RandomVariable(position=1.1, prob=5),
    ]

    reg = MarginalLinearRegression(
        sigma_noise_points=sigma_noise_points, sigma_coef_points=sigma_coef_points
    )
    reg.fit(X, y)

    log_likelihood = []
    for i, sn in enumerate(sigma_noise_points):
        for j, sc in enumerate(sigma_coef_points):
            lr = LinearRegression(sigma_noise=sn.position, sigma_coef=sc.position)
            lr.fit(X=X, y=y)
            assert reg.log_likelihood_over_sigma_[i][j] == lr.log_likelihood_
            log_likelihood.append(lr.log_likelihood_)
    assert reg.log_likelihood_ == logsumexp(log_likelihood) - np.log(4)

    check_linear_model(model=reg, expect_coef=w)

    pred_y = y_scaler.restore(reg.predict(x_scaler.transform(test_X)))
    assert calculate_rmse(test_y, pred_y) <= sigma_noise
