import numpy as np
import pytest
from scipy import integrate

from exhbma import gamma, inverse, uniform


def test_gamma_wo_low_high():
    """
    Test method `gamma` with numerically defined gamma distribution.
    """
    low = -5
    high = 0
    shape = 1e-3
    scale = 1e3
    n_points = 101
    x = np.logspace(low, high, n_points)
    rvs = gamma(x, shape=shape, scale=scale)

    def gamma_distribution(x):
        """Ignoring normalization constant"""
        return x ** (shape - 1) * np.exp(-x / scale)

    const, _ = integrate.quad(gamma_distribution, 10 ** low, 10 ** high)
    distribution = gamma_distribution(x) / const
    for rv, d in zip(rvs, distribution):
        assert rv.prob == pytest.approx(d)


def test_gamma_with_different_low_high():
    """
    Test method `gamma` with numerically defined gamma distribution.
    Set different low and high parameter for `x` and defined range.
    """
    low = -5
    high = 0
    shape = 1e-3
    scale = 1e3
    n_points = 101
    x = np.logspace(low + 1, high - 1, n_points)
    rvs = gamma(x, low=10 ** low, high=10 ** high, shape=shape, scale=scale)

    def gamma_distribution(x):
        """Ignoring normalization constant"""
        return x ** (shape - 1) * np.exp(-x / scale)

    const, _ = integrate.quad(gamma_distribution, 10 ** low, 10 ** high)
    distribution = gamma_distribution(x) / const
    for rv, d in zip(rvs, distribution):
        assert rv.prob == pytest.approx(d)


def test_uniform():
    """
    Test method `uniform` with numerically defined uniform distribution.
    """
    low = 0
    high = 5
    n_points = 101
    x = np.linspace(low, high, n_points)
    rvs = uniform(x, low=low, high=high)

    for rv in rvs:
        assert rv.prob == pytest.approx(1 / (high - low))


def test_inverse_wo_low_high():
    """
    Test method `inverse` with numerically defined gamma distribution.
    """
    low = -5
    high = 0
    n_points = 101
    x = np.logspace(low, high, n_points)
    rvs = inverse(x)

    const = np.log(10 ** high) - np.log(10 ** low)
    distribution = 1 / (x * const)
    for rv, d in zip(rvs, distribution):
        assert rv.prob == pytest.approx(d)


def test_inverse_with_different_low_high():
    """
    Test method `inverse` with numerically defined gamma distribution.
    Set different low and high parameter for `x` and defined range.
    """
    low = -5
    high = 0
    n_points = 101
    x = np.logspace(low + 1, high - 1, n_points)
    rvs = inverse(x, low=10 ** low, high=10 ** high)

    const = np.log(10 ** high) - np.log(10 ** low)
    distribution = 1 / (x * const)
    for rv, d in zip(rvs, distribution):
        assert rv.prob == pytest.approx(d)
