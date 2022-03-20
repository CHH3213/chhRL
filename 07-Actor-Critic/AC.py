# --*--coding: utf-8 --*--
"""
Created with IntelliJ PyCharm.
@Time: 2022/3/20 9:46
@PROJECT_NAME: chhRL
@File: AC.py
@Author: chh3213
@Email:
@Description:
Actor-Critic
    Advantage: AC converge faster than Policy Gradient.
    Disadvantage (IMPORTANT):The Policy is oscillated (difficult to converge), DDPG can solve this problem using advantage of DQN.
    Reference:
        paper: https://papers.nips.cc/paper/1786-actor-critic-algorithms.pdf
        blog:https://blog.csdn.net/weixin_42301220/article/details/123311478
"""

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Actor(nn.Module):
    """
    创建Actor
    """

    def __init__(self, in_dim, out_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        output = F.softmax(self.fc3(x))
        return output


class Critic(nn.Module):
    """
    创建critic
    """

    def __init__(self, in_dim, out_dim=1, hidden_dim=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class AC:
    def __init__(self, state_dim, act_dim, args):
        self.a_log_prob = None
        self.s_dim = state_dim
        self.a_dim = act_dim
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(self.s_dim, self.a_dim, self.args.hidden_dim).to(self.device)
        self.critic = Critic(self.s_dim, 1, self.args.hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.args.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.args.lr_critic)

    def predict(self):
        pass

    def choose_action(self, state):
        """
        根据当前状态选择动作
        :param state:
        :return:
        """
        # print(state)
        state = torch.tensor(state, device=self.device, dtype=torch.float32)
        # state = torch.FloatTensor(state).to(self.device)
        probs = self.actor(state)
        # 分类分布
        m = Categorical(probs)
        # 采样一个action
        action = m.sample()
        self.a_log_prob = m.log_prob(action)
        return action.item()

    def update(self, s, a, r, s_, done):
        s = torch.tensor(s, device=self.device, dtype=torch.float32)
        s_ = torch.tensor(s_, device=self.device, dtype=torch.float32)
        r = torch.tensor(r, device=self.device, dtype=torch.float32)
        v = self.critic(s)
        v_ = self.critic(s_)
        v_target = r+self.args.gamma*v_
        td_error = v_target-v
        a_prob = self.actor(s)[a]
        actor_loss = -(torch.log(a_prob) * td_error.detach())
        # actor_loss = -self.a_log_prob * td_error.detach()
        loss = nn.MSELoss()
        # critic_loss = loss(v_target, v)
        critic_loss = td_error ** 2

        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        critic_loss.backward()

        actor_loss.backward()
        self.critic_optimizer.step()
        self.actor_optimizer.step()

    def save(self, path):
        torch.save(self.actor.state_dict(), path + 'actor.pt')
        torch.save(self.critic.state_dict(), path + 'critic.pt')

    def load(self, path):
        self.actor.load_state_dict(torch.load(path + 'actor.pt'))
        self.critic.load_state_dict(torch.load(path + 'critic.pt'))
