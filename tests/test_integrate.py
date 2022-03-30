import numpy as np
import pytest

from exhbma import (
    integrate_log_values_in_line,
    integrate_log_values_in_square,
    validate_list_dimension,
)


def test_integrate_log_values_in_square():
    """
    Function: f(x, y) = x^2*y^2 + x^2 + y^2
    Integral region: (x, y) in R^2, 0 <= x <= 1, 1 <= y <= 2
    Answer: 31/9
    """

    def func(x, y):
        return x ** 2 * y ** 2 + x ** 2 + y ** 2

    expect = np.log(31 / 9)
    n_x1, n_x2 = 30, 40
    x1 = np.linspace(0, 1, n_x1)
    x2 = np.linspace(1, 2, n_x2)
    log_values = np.log(
        func(x=np.tile(x1.reshape(-1, 1), (1, n_x2)), y=np.tile(x2, (n_x1, 1)))
    )
    result = integrate_log_values_in_square(
        log_values=log_values.tolist(), x1=x1.tolist(), x2=x2.tolist()
    )

    assert result == pytest.approx(expect, rel=1e-3)


def test_integrate_log_values_in_square_with_weights():
    """
    Function: f(x, y) = x^2*y^2 + x^2 + y^2
    Weight: -1
    Integral region: (x, y) in R^2, 0 <= x <= 1, 1 <= y <= 2
    Answer: -31/9
    """

    def func(x, y):
        return x ** 2 * y ** 2 + x ** 2 + y ** 2

    expect = np.log(31 / 9)
    n_x1, n_x2 = 30, 40
    x1 = np.linspace(0, 1, n_x1)
    x2 = np.linspace(1, 2, n_x2)
    log_values = np.log(
        func(x=np.tile(x1.reshape(-1, 1), (1, n_x2)), y=np.tile(x2, (n_x1, 1)))
    )
    weights = -np.ones_like(log_values, dtype=float)
    result = integrate_log_values_in_square(
        log_values=log_values.tolist(),
        x1=x1.tolist(),
        x2=x2.tolist(),
        weights=weights.tolist(),
        expect_positive=False,
    )

    assert result[0] == pytest.approx(expect, rel=1e-3)
    assert result[1] == -1


def test_error_different_shape_integrate_log_values_in_square():
    """
    Test error case of integrate_log_values_in_square
    when shape of log_values and (x1, x2) are different.
    """
    log_values = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    x1 = [0.0, 0.0, 0.0]
    x2 = [0.0, 0.0]

    with pytest.raises(
        ValueError, match=r"Invalid shape, log_values: \(2, 3\), axis: \(3, 2\)"
    ):
        integrate_log_values_in_square(log_values=log_values, x1=x1, x2=x2)


def test_error_different_weight_shape_integrate_log_values_in_square():
    """
    Test error case of integrate_log_values_in_square
    when shape of log_values and weights are different.
    """
    log_values = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    x1 = [0.0, 0.0]
    x2 = [0.0, 0.0, 0.0]
    weights = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]

    with pytest.raises(
        ValueError, match=r"Invalid shape, weights: \(3, 2\), log_values: \(2, 3\)"
    ):
        integrate_log_values_in_square(
            log_values=log_values, x1=x1, x2=x2, weights=weights
        )


def test_error_negative_when_expect_positive_integrate_log_values_in_square():
    """
    Test error case of integrate_log_values_in_square
    when calculation results are negative with `expect_positive=True` option.
    """

    def func(x, y):
        return x ** 2 * y ** 2 + x ** 2 + y ** 2

    n_x1, n_x2 = 30, 40
    x1 = np.linspace(0, 1, n_x1)
    x2 = np.linspace(1, 2, n_x2)
    log_values = np.log(
        func(x=np.tile(x1.reshape(-1, 1), (1, n_x2)), y=np.tile(x2, (n_x1, 1)))
    )
    weights = -np.ones_like(log_values, dtype=float)
    with pytest.raises(ValueError, match="Result is not positive."):
        integrate_log_values_in_square(
            log_values=log_values.tolist(),
            x1=x1.tolist(),
            x2=x2.tolist(),
            weights=weights.tolist(),
            expect_positive=True,
        )


def test_integrate_log_values_in_line():
    """
    Function: f(x, y) = x^2 + x + 1
    Integral region: 1 <= x <= 3
    Answer: 44/3
    """

    def func(x):
        return x ** 2 + x + 1

    expect = np.log(44 / 3)
    n_x1 = 30
    x1 = np.linspace(1, 3, n_x1)
    log_values = np.log(func(x=x1))
    result = integrate_log_values_in_line(
        log_values=log_values.tolist(), x1=x1.tolist()
    )

    assert result == pytest.approx(expect, rel=1e-3)


def test_integrate_log_values_in_line_with_weights():
    """
    Function: f(x, y) = x^2 + x + 1
    Weight: -1
    Integral region: 1 <= x <= 3
    Answer: -44/3
    """

    def func(x):
        return x ** 2 + x + 1

    expect = np.log(44 / 3)
    n_x1 = 30
    x1 = np.linspace(1, 3, n_x1)
    log_values = np.log(func(x=x1))
    weights = -np.ones_like(log_values, dtype=float)
    result = integrate_log_values_in_line(
        log_values=log_values.tolist(),
        x1=x1.tolist(),
        weights=weights.tolist(),
        expect_positive=False,
    )

    assert result[0] == pytest.approx(expect, rel=1e-3)
    assert result[1] == -1


def test_error_different_shape_integrate_log_values_in_line():
    """
    Test error case of integrate_log_values_in_line
    when shape of log_values and (x1, x2) are different.
    """
    log_values = [0.0, 0.0, 0.0]
    x1 = [0.0, 0.0]

    with pytest.raises(
        ValueError, match=r"Invalid shape, log_values: \(3,\), axis: \(2,\)"
    ):
        integrate_log_values_in_line(log_values=log_values, x1=x1)


def test_error_different_weight_shape_integrate_log_values_in_line():
    """
    Test error case of integrate_log_values_in_line
    when shape of log_values and weights are different.
    """
    log_values = [0.0, 0.0, 0.0]
    x1 = [0.0, 0.0, 0.0]
    weights = [0.0, 0.0]

    with pytest.raises(
        ValueError, match=r"Invalid shape, weights: \(2,\), log_values: \(3,\)"
    ):
        integrate_log_values_in_line(log_values=log_values, x1=x1, weights=weights)


def test_error_negative_when_expect_positive_integrate_log_values_in_line():
    """
    Test error case of integrate_log_values_in_line
    when calculation results are negative with `expect_positive=True` option.
    """

    def func(x):
        return x ** 2 + x + 1

    n_x1 = 30
    x1 = np.linspace(1, 3, n_x1)
    log_values = np.log(func(x=x1))
    weights = -np.ones_like(log_values, dtype=float)
    with pytest.raises(ValueError, match="Result is not positive."):
        integrate_log_values_in_line(
            log_values=log_values.tolist(),
            x1=x1.tolist(),
            weights=weights.tolist(),
            expect_positive=True,
        )


def test_validate_list_dimension():
    validate_list_dimension(x=[0, 0, 0], dim=1, name="vector")
    validate_list_dimension(x=[[0, 0, 0], [0, 0, 0]], dim=2, name="matrix")


def test_validate_list_dimension_type():
    with pytest.raises(
        ValueError, match="vector must be list, received `<class 'numpy.ndarray'>`"
    ):
        validate_list_dimension(x=np.array([0, 0, 0]), dim=1, name="vector")


def test_validate_list_dimension_dimension():
    with pytest.raises(ValueError, match="vector must be 1-dim list, received 2-dim"):
        validate_list_dimension(x=[[0, 0, 0], [0, 0, 0]], dim=1, name="vector")

    with pytest.raises(ValueError, match="matrix must be 2-dim list, received 1-dim"):
        validate_list_dimension(x=[0, 0, 0], dim=2, name="matrix")
