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


class Actor_discrete(nn.Module):
    """
    离散动作所用actor，输出分布
    """
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor_discrete, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        # 随机初始化为较小的值
        # init_w = 3e-3
        # self.fc3.weight.data.uniform_(-init_w, init_w)
        # self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        state = F.relu(self.fc1(state))
        state = F.relu(self.fc2(state))
        out = F.softmax(self.fc3(state))
        return out


class Actor_continue(nn.Module):
    """
    对于连续动作，PPO采用的是随机策略，动作基于正态分布进行采样。
    所以Actor网络的目的就是输出正态分布的 mu 和 sigma 。
    """

    def __init__(self, state_dim, action_dim, max_action, hidden_dim):
        super(Actor_continue, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.sigma = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, state):
        state = F.relu(self.fc1(state))
        state = F.relu(self.fc2(state))
        mu = F.tanh(self.mu(state)) * self.max_action
        sigma = F.softplus(self.sigma(state))
        return mu, sigma


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
