# -*- coding: utf-8 -*-

import numpy as np
from .decorator import len_equal


@len_equal
def accuracy_score(y_actual: np.ndarray, y_predict: np.ndarray) -> float:
    """
    计算准确率
    :param y_actual: 真实值
    :param y_predict: 预测值
    :return: 准确率
    """
    if y_actual.shape != y_predict.shape:
        raise ValueError('the shape of y_actual must be equal to y_predict')

    return sum(y_actual == y_predict) / len(y_actual)


@len_equal
def mean_squared_error(y_predict: np.ndarray, y_true: np.ndarray):
    """
    计算y_true和y_predict之间的MSE
    :param y_predict: 预测结果集
    :param y_true: 真值
    :return: MSE
    """
    return np.mean((y_predict - y_true) ** 2)


@len_equal
def root_mean_squared_error(y_predict: np.ndarray, y_true: np.ndarray):
    """
    计算y_true和y_predict之间的RMSE
    :param y_predict: 预测结果集
    :param y_true: 真值
    :return: RMSE
    """
    return np.sqrt(mean_squared_error(y_predict, y_true))


def mean_absolute_error(y_predict: np.ndarray, y_true: np.ndarray):
    """
    计算y_true和y_predict之间的MAE
    :param y_predict: 预测结果集
    :param y_true: 真值
    :return:
    """
    return np.mean(np.absolute(y_predict - y_true))


@len_equal
def r2_score(y_predict: np.ndarray, y_true: np.ndarray):
    """
    计算y_true和y_predict之间的R Square
    :param y_predict: 预测结果集
    :param y_true: 真值
    :return:
    """
    return 1 - mean_squared_error(y_true, y_predict)/np.var(y_true)
