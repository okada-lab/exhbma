import inspect
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from exhbma import ExhaustiveLinearRegression, StandardScaler, gamma


def pytest_convert(obj, expected: bool = False):
    if isinstance(obj, float):
        if expected:
            return pytest.approx(obj)
        else:
            return obj
    elif isinstance(obj, list):
        ret = []
        for item in obj:
            ret.append(pytest_convert(item, expected=expected))
        return ret
    else:
        return obj


def check_dict(obj: Dict, expected: Dict):
    for key in obj.keys():
        if isinstance(obj[key], dict):
            check_dict(obj[key], expected[key])
        else:
            assert pytest_convert(obj[key]) == pytest_convert(
                expected[key], expected=True
            ), f"Test failed in {key}"


@pytest.mark.tutorial
def test_exhbma_with_linear_regression(force_update: bool):
    """
    Test tutorial code
    """
    self_func_name = inspect.currentframe().f_code.co_name
    cache_dir = Path(__file__).parent / "cache"
    cache_dir.mkdir(exist_ok=True)
    cache_file_path = cache_dir / f"{self_func_name}.json"
    if cache_file_path.exists():
        with open(cache_file_path) as f:
            prev_result = json.load(f)
    else:
        prev_result = {}

    # Generate sample dataset
    n_data, n_features = 50, 10
    sigma_noise = 0.1

    nonzero_w = [1, 1, -0.8, 0.5]
    w = nonzero_w + [0] * (n_features - len(nonzero_w))

    np.random.seed(0)
    X = np.random.randn(n_data, n_features)
    y = np.dot(X, w) + sigma_noise * np.random.randn(n_data)

    # Train a model
    # Data preprocessing
    x_scaler = StandardScaler(n_dim=2)
    y_scaler = StandardScaler(n_dim=1, scaling=False)
    x_scaler.fit(X)
    y_scaler.fit(y)
    X = x_scaler.transform(X)
    y = y_scaler.transform(y)

    # Model fitting
    n_sigma_points = 20
    sigma_noise_log_range = [-2.5, 0.5]
    sigma_coef_log_range = [-2, 1]
    reg = ExhaustiveLinearRegression(
        sigma_noise_points=gamma(
            np.logspace(
                sigma_noise_log_range[0], sigma_noise_log_range[1], n_sigma_points
            ),
        ),
        sigma_coef_points=gamma(
            np.logspace(
                sigma_coef_log_range[0], sigma_coef_log_range[1], n_sigma_points
            ),
        ),
    )
    reg.fit(X, y)

    # Results
    result = {}

    # Feature posterior
    result["feature_posteriors_"] = reg.feature_posteriors_

    # Sigma posterior distribution
    result["log_likelihood_over_sigma_"] = reg.log_likelihood_over_sigma_

    # Coefficient
    result["coef_"] = reg.coef_

    # Weight diagram
    result["models_"] = []
    for m in reg.models_:
        result["models_"].append(m.coefficient)

    result["log_likelihoods_"] = reg.log_likelihoods_
    result["log_likelihood_"] = reg.log_likelihood_

    # Prediction for new data
    n_test = 10 ** 3

    test_X = np.random.randn(n_test, n_features)
    test_y = np.dot(test_X, w) + sigma_noise * np.random.randn(n_test)

    pred_y = y_scaler.restore(reg.predict(x_scaler.transform(test_X), mode="full"))
    rmse = np.power(test_y - pred_y, 2).mean() ** 0.5
    result["rmse"] = rmse

    if (not cache_file_path.exists()) or force_update:
        with open(cache_file_path, "w") as f:
            json.dump(result, fp=f)

    check_dict(obj=result, expected=prev_result)
