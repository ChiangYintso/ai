# -*- coding: utf-8 -*-
from collections import Counter
import numpy as np
from typing import Optional
from jyzml.utils.metrics import accuracy_score
from enum import Enum


class KNNClassifier:
    """
    KNN分类器
    """

    class WEIGHT(Enum):
        """
        KNN weight超参数枚举
        """
        UNIFORM = 'uniform'
        DISTANCE = 'distance'

    def __init__(self, k: int = 5, weight: WEIGHT = WEIGHT.UNIFORM, p: int = 2):
        """
        构造方法
        :param k: KNN的超参数k, 经验上最佳值是5
        :param weight: KNN的投票权重超参数
        :param p: 明可夫斯基距离超参数
        """
        if k < 1:
            raise ValueError('param n must large than 0')
        self.__k: int = k
        self.__X_train: Optional[np.ndarray, None] = None
        self.__y_train: Optional[np.ndarray, None] = None
        self.__weight: KNNClassifier.WEIGHT = weight
        self.__p: int = p

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """根据数据集训练KNN"""
        self.__X_train = X_train
        self.__y_train = y_train

    def predict(self, X_predict: np.ndarray) -> np.ndarray:
        """给定待测数据集求结果"""
        if self.__X_train is None or self.__y_train is None:
            raise ValueError('must fit before predict')

        if X_predict is None:
            raise ValueError('X_predict must not be None')

        y_predict = [self.__predict(x) for x in X_predict]
        return np.array(y_predict, dtype=self.__y_train.dtype)

    def __predict(self, x_predict: np.ndarray):
        distances = [np.sum((x_train - x_predict) ** self.__p) ** (1 / self.__p) for x_train in self.__X_train]
        nearest: np.ndarray = np.argsort(distances)
        top_k_y = [self.__y_train[i] for i in nearest[:self.__k]]
        votes = Counter(top_k_y)
        return votes.most_common(1)[0][0]

    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        根据测试数据集计算准确度
        :param X_test: 测试特征矩阵
        :param y_test: 测试标记向量
        :return: 准确率
        """
        y_predict = self.predict(X_test)
        return accuracy_score(y_actual=y_test, y_predict=y_predict)

    def __repr__(self):
        return 'KNN(k=%d)' % self.__k
