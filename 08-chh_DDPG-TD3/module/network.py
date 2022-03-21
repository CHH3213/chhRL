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


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action
        # 随机初始化为较小的值
        # init_w = 3e-3
        # self.fc3.weight.data.uniform_(-init_w, init_w)
        # self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        state = F.relu(self.fc1(state))
        state = F.relu(self.fc2(state))
        out = self.max_action * torch.tanh(self.fc3(state))
        return out


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
