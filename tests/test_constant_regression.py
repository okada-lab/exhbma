import numpy as np
import pytest

from exhbma import ConstantRegression, StandardScaler


@pytest.fixture()
def seed():
    val = 0
    np.random.seed(val)
    return val


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
