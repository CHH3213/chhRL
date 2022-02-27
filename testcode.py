# --*--coding: utf-8 --*--
"""
Created with IntelliJ PyCharm.
@Time: 2022/2/26 9:29
@PROJECT_NAME: chhRL
@File: testcode.py
@Author: chh3213
@Email:
@Description:
测试python语法代码，与强化学习无关
"""
from collections import defaultdict
import numpy as np

# for t in range(100, -1, -1):
#     print(t)


Q = defaultdict(lambda: np.zeros(4))
Q[0][1]=2
Q[0][0]=3
Q[0][3]=5
print(np.argmax(Q[0]))


print(np.eye(4))

print(np.ones([16, 4]) / 4)
policy = np.ones([16, 4]) / 4


