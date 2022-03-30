import numpy as np


class StandardScaler(object):
    def __init__(self, n_dim, scaling: bool = True):
        self.n_dim = n_dim
        self.scaling = scaling

    def fit(self, array):
        if self.n_dim == 1:
            self.mean = np.mean(array)
            self.std = np.mean((array - self.mean) ** 2) ** 0.5
        elif self.n_dim == 2:
            self.mean = np.mean(array, axis=0)
            self.std = np.mean((array - self.mean) ** 2, axis=0) ** 0.5

    def transform(self, array):
        ret = np.array(array) - self.mean
        if self.scaling:
            ret = ret / self.std
        return ret

    def restore(self, array):
        ret = np.array(array)
        if self.scaling:
            ret = ret * self.std
        ret = ret + self.mean
        return ret
