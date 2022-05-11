from typing import List

import numpy as np
import pytest

from exhbma import ExhaustiveLinearRegression, StandardScaler, gamma


@pytest.fixture()
def seed():
    val = 0
    np.random.seed(val)
    return val


def check_basic_attribute_after_fit(model: ExhaustiveLinearRegression, n_features: int):
    assert model.n_features_in_ == n_features
    assert len(model.indicators_) == 2 ** n_features


def check_feature_posteriors(
    model: ExhaustiveLinearRegression, n_features: int, nonzero_index: List[bool]
):
    assert len(model.feature_posteriors_) == n_features
    assert np.all(np.array(model.feature_posteriors_)[nonzero_index] > 0.9)
    assert np.all(
        np.array(model.feature_posteriors_)[np.logical_not(nonzero_index)] < 0.1
    )


def check_linear_model(model: ExhaustiveLinearRegression, expect_coef: np.ndarray):
    assert model.coef_ == pytest.approx(expect_coef, rel=1e-1, abs=1e-3)


def calculate_rmse(true_y, pred_y) -> float:
    return np.power(true_y - pred_y, 2).mean() ** 0.5


def check_prediction(
    model: ExhaustiveLinearRegression,
    test_X,
    test_y,
    x_scaler,
    y_scaler,
    precision: float,
):
    pred_y = y_scaler.restore(model.predict(x_scaler.transform(test_X), mode="full"))
    assert calculate_rmse(test_y, pred_y) <= precision

    pred_y = y_scaler.restore(model.predict(x_scaler.transform(test_X), mode="select"))
    assert calculate_rmse(test_y, pred_y) <= precision


def test_exhaustive_linear_regression(seed):
    """
    Test method `fit`.
    """
    n_data, n_features = 200, 8
    n_test = 1000
    sigma_noise = 0.01
    alpha = 0.5
    prob_data_points = 10

    X = np.random.randn(n_data, n_features)
    nonzero_coefs = [1, 0.8, 0.5]
    w = np.array(nonzero_coefs + [0] * (n_features - len(nonzero_coefs)))
    y = np.dot(X, w) + np.random.randn(n_data) * sigma_noise
    test_X = np.random.randn(n_test, n_features)
    test_y = np.dot(test_X, w)

    x_scaler = StandardScaler(n_dim=2)
    y_scaler = StandardScaler(n_dim=1, scaling=False)
    x_scaler.fit(X)
    y_scaler.fit(y)
    X = x_scaler.transform(X)
    y = y_scaler.transform(y)

    reg = ExhaustiveLinearRegression(
        sigma_noise_points=gamma(
            np.logspace(-3, 0, prob_data_points), low=1e-3, high=1e0
        ),
        sigma_coef_points=gamma(
            np.logspace(-2, 1, prob_data_points), low=1e-2, high=1e1
        ),
        alpha=alpha,
    )
    reg.fit(X, y)

    check_basic_attribute_after_fit(model=reg, n_features=n_features)

    check_feature_posteriors(
        model=reg,
        n_features=n_features,
        nonzero_index=[True, True, True] + [False] * (n_features - len(nonzero_coefs)),
    )

    check_linear_model(model=reg, expect_coef=w)

    check_prediction(
        model=reg,
        test_X=test_X,
        test_y=test_y,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        precision=sigma_noise,
    )


def test_exhaustive_linear_regression_case_different_sigma_points(seed):
    """
    Test method `fit` when different sigma_points are passed.
    """
    n_data, n_features = 200, 8
    n_test = 1000
    sigma_noise = 0.01
    alpha = 0.5
    prob_data_points = 10

    X = np.random.randn(n_data, n_features)
    nonzero_coefs = [1, 0.8, 0.5]
    w = np.array(nonzero_coefs + [0] * (n_features - len(nonzero_coefs)))
    y = np.dot(X, w) + np.random.randn(n_data) * sigma_noise
    test_X = np.random.randn(n_test, n_features)
    test_y = np.dot(test_X, w)

    x_scaler = StandardScaler(n_dim=2)
    y_scaler = StandardScaler(n_dim=1, scaling=False)
    x_scaler.fit(X)
    y_scaler.fit(y)
    X = x_scaler.transform(X)
    y = y_scaler.transform(y)

    reg = ExhaustiveLinearRegression(
        sigma_noise_points=gamma(
            np.logspace(-3, 0, prob_data_points + 3), low=1e-3, high=1e0
        ),
        sigma_coef_points=gamma(
            np.logspace(-2, 1, prob_data_points), low=1e-2, high=1e1
        ),
        alpha=alpha,
    )
    reg.fit(X, y)

    check_basic_attribute_after_fit(model=reg, n_features=n_features)

    check_feature_posteriors(
        model=reg,
        n_features=n_features,
        nonzero_index=[True, True, True] + [False] * (n_features - len(nonzero_coefs)),
    )

    check_linear_model(model=reg, expect_coef=w)

    check_prediction(
        model=reg,
        test_X=test_X,
        test_y=test_y,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        precision=sigma_noise,
    )


def test_generate_indicator():
    """
    Test method `_generate_indicator` with manually generated indicators.
    """
    reg = ExhaustiveLinearRegression(
        sigma_noise_points=[],
        sigma_coef_points=[],
        alpha=0.5,
    )
    n_features = 3
    indicators = reg._generate_indicator(n_features=n_features)
    expect = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]
    assert indicators == expect


def test_generate_indicator_excluding_null():
    """
    Test method `_generate_indicator` with manually generated indicators.
    """
    reg = ExhaustiveLinearRegression(
        sigma_noise_points=[], sigma_coef_points=[], alpha=0.5, exclude_null=True
    )
    n_features = 3
    indicators = reg._generate_indicator(n_features=n_features)
    expect = [
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]
    assert indicators == expect


def test_fixed_alpha_prior():
    """
    Test method `_fixed_alpha_prior` with manually calculated values.
    """
    reg = ExhaustiveLinearRegression(
        sigma_noise_points=[],
        sigma_coef_points=[],
        alpha=0.5,
    )
    indicators = [
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]
    expect = 1 / 7
    for indicator in indicators:
        assert reg._fixed_alpha_prior(indicator=indicator) == pytest.approx(
            np.log(expect)
        )


def test_fixed_alpha_prior_not_half():
    """
    Test method `_fixed_alpha_prior` with manually calculated values
    when alpha is not 0.5.
    """
    reg = ExhaustiveLinearRegression(
        sigma_noise_points=[],
        sigma_coef_points=[],
        alpha=0.8,
    )
    indicators = [
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]
    probs = [0.032, 0.032, 0.128, 0.032, 0.128, 0.128, 0.512]
    norm = sum(probs)
    probs = [p / norm for p in probs]
    for indicator, p in zip(indicators, probs):
        assert reg._fixed_alpha_prior(indicator=indicator) == pytest.approx(np.log(p))
