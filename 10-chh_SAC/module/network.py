#!/usr/bin/python3
# --*--coding:utf-8--*--
"""
==========chhRL===============
@File: network.py
@Time: 2022/3/3 下午8:58
@Author: chh3213
@Description:

========Above the sun, full of fire!=============
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import math


class Stochastic_Actor(nn.Module):
    """
    对于连续动作，SAC采用的是随机策略，动作基于正态分布进行采样。
    所以Actor网络的目的就是输出正态分布的 mu 和 sigma 。
    """

    def __init__(self, state_dim, action_dim, hidden_dim, init_w=3e-3, min_log_std=-20, max_log_std=2):
        super(Stochastic_Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.sigma = nn.Linear(hidden_dim, action_dim)

        # 随机初始化为较小的权重值
        self.mu.weight.data.uniform_(-init_w, init_w)
        self.mu.bias.data.uniform_(-init_w, init_w)
        self.sigma.weight.data.uniform_(-init_w, init_w)
        self.sigma.bias.data.uniform_(-init_w, init_w)

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, state):
        state = F.relu(self.fc1(state))
        state = F.relu(self.fc2(state))
        mu = self.mu(state)
        log_sigma = torch.clamp(self.sigma(state), self.min_log_std, self.max_log_std)
        return mu, log_sigma


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        # 随机初始化为较小的值
        # init_w = 3e-3
        # self.fc3.weight.data.uniform_(-init_w, init_w)
        # self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        q = F.relu(self.fc1(torch.cat([state, action], 1)))
        q = F.relu(self.fc2(q))
        out = self.fc3(q)
        return out


class V_Net(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(V_Net, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        # 随机初始化为较小的值
        # init_w = 3e-3
        # self.fc3.weight.data.uniform_(-init_w, init_w)
        # self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        state = F.relu(self.fc1(state))
        state = F.relu(self.fc2(state))
        out = self.fc3(state)
        return out
