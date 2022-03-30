from typing import List, Optional

import numpy as np
from scipy.special import logsumexp


def integrate_log_values_in_square(
    log_values: List[List[float]],
    x1: List[float],
    x2: List[float],
    weights: Optional[List[List[float]]] = None,
    expect_positive: bool = True,
):
    r"""
    Integrate box region.
    \int weight * exp(log_value) dx1 dx2

    log_values: 2 dimension array with shape (len(x1), len(x2))
        Log values to be integrated over x1 and x2 axes.

    x1: List[float]
        1st axis points.

    x2: List[float]
        2nd axis points.

    weights: Optional[List[List[float]]], default: None
        Weights to log_values.

    expect_positive: bool, default: True
        If set True, ValueError is raised when result is not positive.
    """
    validate_list_dimension(x1, dim=1, name="x1")
    validate_list_dimension(x2, dim=1, name="x2")

    np_log_values = np.array(log_values)
    if np_log_values.shape != (len(x1), len(x2)):
        raise ValueError(
            "Invalid shape, log_values: {}, axis: {}".format(
                np_log_values.shape, (len(x1), len(x2))
            )
        )

    if weights is None:
        np_weights = np.ones_like(np_log_values, dtype=float)
    else:
        np_weights = np.array(weights)
        if np_weights.shape != np_log_values.shape:
            raise ValueError(
                "Invalid shape, weights: {}, log_values: {}".format(
                    np_weights.shape, np_log_values.shape
                )
            )

    val = np_log_values - np.log(4)

    # For ease of implementation, extend axis points with edge values.
    extended_x1 = np.array([x1[0]] + x1 + [x1[-1]])
    extended_x2 = np.array([x2[0]] + x2 + [x2[-1]])
    area = np.zeros_like(np_log_values, dtype=float)
    for i in range(2):
        for j in range(2):
            area += np.outer(
                extended_x1[i + 1 : i + 1 + len(x1)] - extended_x1[i : i + len(x1)],
                extended_x2[j + 1 : j + 1 + len(x2)] - extended_x2[j : j + len(x2)],
            )
    val += np.log(area)

    result = logsumexp(val, b=np_weights, return_sign=True)
    if expect_positive:
        if result[1] <= 0:
            raise ValueError("Result is not positive.")
        return result[0]
    else:
        """(log of calculation, sign)"""
        return result


def integrate_log_values_in_line(
    log_values: List[float],
    x1: List[float],
    weights: Optional[List[float]] = None,
    expect_positive: bool = True,
):
    """
    Integrate along line.

    log_values: 1 dimension array with length len(x1)
        Log values to be integrated along x1

    x1: List[float]
        1st axis points.

    weights: Optional[List[float]], default: None
        Weights to log_values.

    expect_positive: bool, default: True
        If set True, ValueError is raised when result is not positive.
    """
    validate_list_dimension(x1, dim=1, name="x1")

    np_log_values = np.array(log_values)
    if np_log_values.shape != (len(x1),):
        raise ValueError(
            "Invalid shape, log_values: {}, axis: {}".format(
                np_log_values.shape, (len(x1),)
            )
        )

    if weights is None:
        np_weights = np.ones_like(np_log_values, dtype=float)
    else:
        np_weights = np.array(weights)
        if np_weights.shape != np_log_values.shape:
            raise ValueError(
                "Invalid shape, weights: {}, log_values: {}".format(
                    np_weights.shape, np_log_values.shape
                )
            )

    val = np_log_values - np.log(2)

    # For ease of implementation, extend axis points with edge values.
    extended_x1 = np.array([x1[0]] + x1 + [x1[-1]])
    area = np.zeros_like(np_log_values, dtype=float)
    for i in range(2):
        area += extended_x1[i + 1 : i + 1 + len(x1)] - extended_x1[i : i + len(x1)]
    val += np.log(area)

    result = logsumexp(val, b=np_weights, return_sign=True)
    if expect_positive:
        if result[1] <= 0:
            raise ValueError("Result is not positive.")
        return result[0]
    else:
        """(log of calculation, sign)"""
        return result


def validate_list_dimension(x, dim: int, name: str):
    if not isinstance(x, list):
        raise ValueError("{} must be list, received `{}`".format(name, type(x)))

    np_x = np.array(x)
    if len(np_x.shape) != dim:
        raise ValueError(
            "{} must be {}-dim list, received {}-dim".format(name, dim, len(np_x.shape))
        )
