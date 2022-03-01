# --*--coding: utf-8 --*--
"""
Created with IntelliJ PyCharm.
@Time: 2022/2/28 16:45
@PROJECT_NAME: chhRL
@File: REINFORCE_discrete.py
@Author: chh3213
@Email:
@Description:
The policy gradient algorithm works by updating policy parameters via stochastic gradient ascent on policy performance.
It's an on-policy algorithm can be used for environments with either discrete or continuous action spaces.
Here is an example on discrete action space game CartPole-v0.
To apply it on continuous action space, you need to change the last softmax layer and the choose_action function.

reference:
        https://blog.csdn.net/weixin_42301220/article/details/123112474
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.129.8871&rep=rep1&type=pdf
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.distributions import Categorical
from torch.autograd import Variable


class Network(nn.Module):
    """
    定义一个简单的全连接层网络
    """

    def __init__(self, input_dim, output_dim, hidden_dim):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = F.softmax(self.fc3(x))
        return output


class REINFORCEPolicy:
    def __init__(self, state_dim, action_dim, hidden_dim, args):
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = args.lr
        self.gamma = args.gamma
        self.log_probs = []
        # 一个浮点数
        self.eps = np.finfo(np.float32).eps.item()
        # 如果有gpu则使用gpu
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = Network(self.s_dim, self.a_dim, hidden_dim).to(self.device)
        print(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def choose_action(self, state):
        """
        根据当前状态选择动作
        :param state:
        :return:
        """
        # print(state)
        state = torch.from_numpy(state).float()
        state = state.to(self.device)
        state = Variable(state)
        probs = self.net(state)
        # 分类分布
        m = Categorical(probs)
        # 采样一个action
        action = m.sample()
        # m.log_prob(action) 就是 logp
        self.log_probs.append(m.log_prob(action))
        return action.item()

    def predict(self):
        pass

    def update(self, rewards_episode, states_episode):
        """
        更新网络
        :param rewards_episode: 一个episode的奖励，计算总回报用
        :param states_episode: 一个episode的状态,这边没用到，
        因为计算log_probs在choose_action函数中进行了，所以不需要
        :return:
        """
        # 总收益
        R = 0
        rewards = []
        loss_sum = 0
        for r in rewards_episode[::-1]:
            # r为每一步的奖励
            if r == 0:
                R = 0
            else:
                R = r + self.gamma * R

            rewards.insert(0, R)
        # 对回报进行normalize
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps)
        self.optimizer.zero_grad()
        for log_prob, reward in zip(self.log_probs, rewards):
            # 计算loss
            loss = -log_prob * reward
            loss_sum += loss
            # 每步计算梯度
            # 反向传播计算梯度
            loss.backward()
        # 一个episode的所有loss相加再计算梯度,效果更差
        # loss_sum.backward()
        # 更新权重
        self.optimizer.step()
        self.log_probs = []

    def save(self, path):
        torch.save(self.net.state_dict(), path + 'model.pt')

    def load(self, path):
        self.net.load_state_dict(torch.load(path + 'model.pt'))
