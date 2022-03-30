import numpy as np
import pytest

from exhbma import StandardScaler


def test_1_dimension():
    """
    Test array is 1-dimension case.
    """
    scaler = StandardScaler(n_dim=1, scaling=True)
    X = np.array([1, 5, 9])
    scaler.fit(X)

    expected_mean = 5
    expected_std = np.sqrt(32 / 3)
    assert scaler.mean == pytest.approx(expected_mean)
    assert scaler.std == pytest.approx(expected_std)

    assert scaler.transform(X) == pytest.approx(
        np.array([-4 / expected_std, 0, 4 / expected_std])
    )

    assert scaler.restore(scaler.transform(X)) == pytest.approx(X)


def test_2_dimension():
    """
    Test array is 2-dimension case.
    """
    scaler = StandardScaler(n_dim=2, scaling=True)
    X = np.array([[1, 2], [3, 8], [5, 14]])
    scaler.fit(X)

    expected_mean = np.array([3, 8])
    expected_std = np.array([np.sqrt(8 / 3), np.sqrt(24)])
    assert scaler.mean == pytest.approx(expected_mean)
    assert scaler.std == pytest.approx(expected_std)

    assert scaler.transform(X) == pytest.approx(
        np.array(
            [
                [-2 / expected_std[0], -6 / expected_std[1]],
                [0, 0],
                [2 / expected_std[0], 6 / expected_std[1]],
            ]
        )
    )

    assert scaler.restore(scaler.transform(X)) == pytest.approx(X)


def test_wo_scaling():
    """
    Test option scaling == False case.
    """
    scaler = StandardScaler(n_dim=2, scaling=False)
    X = np.array([[1, 2], [3, 8], [5, 14]])
    scaler.fit(X)

    expected_mean = np.array([3, 8])
    expected_std = np.array([np.sqrt(8 / 3), np.sqrt(24)])
    assert scaler.mean == pytest.approx(expected_mean)
    assert scaler.std == pytest.approx(expected_std)

    assert scaler.transform(X) == pytest.approx(np.array([[-2, -6], [0, 0], [2, 6]]))

    assert scaler.restore(scaler.transform(X)) == pytest.approx(X)
