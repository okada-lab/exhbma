import numpy as np


class StandardScaler:
    def __init__(self, n_dim: int, scaling: bool = True) -> None:
        self.n_dim = n_dim
        self.scaling = scaling
        self.mean: float | np.ndarray
        self.std: float | np.ndarray

    def fit(self, array: np.ndarray) -> None:
        if self.n_dim == 1:
            self.mean = float(np.mean(array))
            self.std = float(np.mean((array - self.mean) ** 2) ** 0.5)
        elif self.n_dim == 2:
            self.mean = np.mean(array, axis=0)
            self.std = np.mean((array - self.mean) ** 2, axis=0) ** 0.5

    def transform(self, array: np.ndarray) -> np.ndarray:
        ret = np.array(array) - self.mean
        if self.scaling:
            ret = ret / self.std
        return ret

    def restore(self, array: np.ndarray) -> np.ndarray:
        ret = np.array(array)
        if self.scaling:
            ret = ret * self.std
        ret = ret + self.mean
        return ret
