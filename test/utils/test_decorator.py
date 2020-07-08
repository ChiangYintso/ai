# -*- coding: utf-8 -*-

from jyzml.utils.decorator import len_equal


def test_len_equal():

    @len_equal
    def foo(a, b, c=None):
        """
        测试函数
        :param a: 任意参数
        :param b: 任意参数
        :param c: 任意参数
        """
        if c is None:
            c = [1, 2]
        print('hello')
        print('world')
        print(a, b, c)

    foo([2, ], [2, ], [3, ])
