# -*- coding: utf-8 -*-
from functools import wraps


def len_equal(func):
    """
    校验函数参数长度相等
    :param func: 函数
    :return: 函数
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        """
        校验函数参数是否相等
        :param args: 参数
        :param kwargs: 参数
        :return: func(args, kwargs)
        """
        length = len(args[0])
        for arg in args:
            if len(arg) != length:
                raise ValueError('length of args must be equal')
        for kwarg in kwargs:
            if len(kwargs[kwarg]) != length:
                raise ValueError('length of args must be equal')
        return func(args, kwargs)

    return wrapper
