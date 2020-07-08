# -*- coding: utf-8 -*-
import numpy as np


class StandardScaler:

    def __init__(self):
        self.__mean = None
        self.__scale = None

    def fit(self, X: np.ndarray):
        if X.ndim != 2:
            raise ValueError('The dimension of X must be 2')

        self.__mean = np.array([np.mean(X[:, i]) for i in range(X.shape[1])])
        self.__scale = np.array([np.std(X[:, i]) for i in range(X.shape[1])])

        return self

    def transform(self, X):
        if X.ndim != 2:
            raise ValueError('The dimension of X must be 2')

        if self.__mean is None:
            raise ValueError('must fit before transform')

        if len(self.__mean) != X.shape[1]:
            raise ValueError('the feature number of X must be equal to mean_ and std_')
        result_X = np.empty(shape=X.shape, dtype=float)
        for col in range(X.shape[1]):
            result_X[:, col] = (X[:, col] - self.__mean[col]) / self.__scale[col]

        return result_X

    @property
    def mean_(self):
        return self.__mean

    @property
    def scale_(self):
        return self.__scale
