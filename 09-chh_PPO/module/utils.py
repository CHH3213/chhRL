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
import gym


def soft_update(target, source, tau=0.001):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class NormalizedActions(gym.ActionWrapper):
    """ 
    将action范围重定在[0.1]之间
    """

    def action(self, action):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high
        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        action = np.clip(action, low_bound, upper_bound)
        return action

    def reverse_action(self, action):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high
        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        action = np.clip(action, low_bound, upper_bound)
        return action


