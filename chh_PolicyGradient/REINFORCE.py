# --*--coding: utf-8 --*--
"""
Created with IntelliJ PyCharm.
@Time: 2022/2/28 16:45
@PROJECT_NAME: chhRL
@File: REINFORCE.py
@Author: chh3213
@Email:
@Description:
The policy gradient algorithm works by updating policy parameters via stochastic gradient ascent on policy performance.
It's an on-policy algorithm can be used for environments with either discrete or continuous action spaces.
Here is an example on discrete action space game CartPole-v0.
To apply it on continuous action space, you need to change the last softmax layer and the choose_action function.
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
        self.eps = np.finfo(np.float32).eps.item()
        self.net = Network(self.s_dim, self.a_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def choose_action(self, state):
        # print(state)
        state = torch.from_numpy(state).float()
        state = Variable(state)
        probs = self.net(state)
        m = Categorical(probs)
        action = m.sample()
        # print(action.item())
        self.log_probs.append(m.log_prob(action))
        return action.item()

    def predict(self):
        pass

    def update(self,rewards_list):
        R = 0
        rewards = []
        for r in rewards_list[::-1]:
            if r==0:
                R=0
            else:
                R = r + self.gamma*R

            rewards.insert(0, R)
        # Normalize reward
        rewards = torch.tensor(rewards)
        rewards = (rewards-rewards.mean())/(rewards.std()+self.eps)
        self.optimizer.zero_grad()
        for log_prob, reward in zip(self.log_probs, rewards):
            loss = -log_prob*reward

            loss.backward()

        self.optimizer.step()
        self.log_probs = []

    def save(self, path):
        torch.save(self.net.state_dict(),path+'model.pt')

    def load(self, path):
        self.net.load_state_dict(torch.load(path+'model.pt'))
