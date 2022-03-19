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


class DuelingNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(DuelingNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # 优势函数
        self.advantage = nn.Linear(hidden_dim, output_dim)
        # 价值函数
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        advantage = self.advantage(x)
        value = self.value(x)
        # output = value + advantage
        # 论文中给的两种方式,不过在本次实验中，直接上面的输出结果比下面两种好
        # output = value+advantage-advantage.max()
        output = value + advantage - advantage.mean()  # 在实际应用中，效果更好。
        return output


class NoisyNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(NoisyNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.fc3 = NoisyLinear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out


class NoisyLinear(nn.Linear):
    # Noisy Linear Layer for independent Gaussian Noise
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        # make the sigmas trainable:
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        # not trainable tensor for the nn.Module
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        # extra parameter for the bias and register buffer for the bias parameter
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))

        # reset parameter as initialization of the layer
        self.reset_parameter()

    def reset_parameter(self):
        """
        initialize the parameter of the layer and bias
        """
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):
        # sample random noise in sigma weight buffer and bias buffer
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias
        return F.linear(input, self.weight + self.sigma_weight * self.epsilon_weight, bias)


class NoisyLinear2(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear2, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(Variable(self.weight_epsilon))
            bias = self.bias_mu + self.bias_sigma.mul(Variable(self.bias_epsilon))
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x


class DistributionalNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_atoms):
        super(DistributionalNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_atoms = num_atoms
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim * num_atoms)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        out = F.softmax(x.view(-1, self.output_dim, self.num_atoms), dim=2)
        return out


class RainbowNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(RainbowNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # 优势函数
        self.advantage = nn.Linear(hidden_dim, output_dim)
        # 价值函数
        self.value = nn.Linear(hidden_dim, 1)
        # self.fc3 = NoisyLinear(hidden_dim, output_dim)
        # self.fc3 = nn.Linear(hidden_dim,output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        advantage = self.advantage(x)
        value = self.value(x)
        output = value + advantage - advantage.mean()
        # output = self.fc3(x)
        # 论文中给的两种方式,不过在本次实验中，直接上面的输出结果比下面两种好
        # output = value+advantage-advantage.max()
        # output = value+advantage-advantage.mean()  # 在实际应用中，效果更好。
        return output
