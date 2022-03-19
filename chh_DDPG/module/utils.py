#!/usr/bin/python3
# --*--coding:utf-8--*--
"""
==========chhRL===============
@File: utils.py
@Time: 2022/3/3 下午2:12
@Author: chh3213
@Description:
目标网络更新方式:硬更新，软更新

========Above the sun, full of fire!=============
"""
import numpy as np
import random
import torch
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
        
        
