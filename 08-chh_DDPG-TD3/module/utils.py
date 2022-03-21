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


class OUNoise(object):
    """
    Ornstein–Uhlenbeck噪声
    """

    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.obs = None
        self.mu = mu  # OU噪声的参数
        self.theta = theta  # OU噪声的参数
        self.sigma = max_sigma  # OU噪声的参数
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        self.obs = np.ones(self.action_dim) * self.mu

    def evolve_obs(self):
        x = self.obs
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.obs = x + dx
        return self.obs

    def get_action(self, action, t=0):
        ou_obs = self.evolve_obs()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)  # sigma会逐渐衰减
        return np.clip(action + ou_obs, self.low, self.high)  # 动作加上噪声后进行剪切
