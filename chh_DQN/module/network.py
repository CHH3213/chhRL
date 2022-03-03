#!/usr/bin/python3
# --*--coding:utf-8--*--
"""
==========chhRL===============
@File: network.py
@Time: 2022/3/3 下午8:58
@Author: chh3213
@Description:
神经网络模块，包含一个全连接网络，一个CNN网络

========Above the sun, full of fire!=============
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable
import random

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out


class CNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.features = nn.Sequential(
            nn.Conv2d(input_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.output_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_dim))).view(1, -1).size(1)

    # def act(self, state, epsilon):
    #     if random.random() > epsilon:
    #         state = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), volatile=True)
    #         q_value = self.forward(state)
    #         action = q_value.max(1)[1].data[0]
    #     else:
    #         action = random.randrange(env.action_space.n)
    #     return action
