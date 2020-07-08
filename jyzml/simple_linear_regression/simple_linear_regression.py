# -*- coding: utf-8 -*-
"""
简单线性回归类
"""
import numpy as np
from jyzml.utils.metrics import r2_score


class SimpleLinearRegression:
    """
    简单线性回归
    """

    def __init__(self):
        self.__a = None
        self.__b = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        """
        根据训练数据集x_train和y_train训练简单线性回归模型
        :param x_train: 训练数据集
        :param y_train: 训练数据结果集
        :return: self
        """
        if x_train.ndim != 1:
            raise ValueError('Simple Linear Regression can only solve single feature training data')

        if len(x_train) != len(y_train):
            raise ValueError('the size of x_train must be equal to the size of y_train')

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        # 向量化运算
        w: np.array = x_train - x_mean
        num = w.dot(y_train - y_mean)
        d = w.dot(w)

        self.__a = num / d
        self.__b = y_mean - self.__a * x_mean

        return self

    def predict(self, x_predict: np.ndarray) -> np.ndarray:
        """
        给定待预测数据集x_predict，返回结果向量
        :param x_predict: 待预测数据集
        :return: 结果向量
        """
        if self.__a is None:
            raise RuntimeError('must fit before predict')

        if x_predict.ndim != 1:
            raise ValueError('Simple Linear Regression can only solve single feature training data')

        return np.array([self.__predict(x) for x in x_predict])

    def __predict(self, x_single):
        return self.__a * x_single + self.__b

    def score(self, x_test: np.ndarray, y_test: np.ndarray):
        """
        根据测试数据集 x_test 和 y_test 确定当前模型的准确度
        :param x_test: 测试数据集
        :param y_test: 测试结果集
        :return: 在[0, 1]范围内的浮点数
        """
        y_predict = self.predict(x_test)
        return r2_score(y_predict, y_test)

    @property
    def a_(self):
        """
        线性回归参数a
        :return: a
        """
        return self.__a

    @property
    def b_(self):
        """
        线性回归参数b
        :return: b
        """
        return self.__b

    def __repr__(self):
        return self.__class__.__name__ + '()'
