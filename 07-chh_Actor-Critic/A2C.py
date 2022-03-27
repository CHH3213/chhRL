# --*--coding: utf-8 --*--
"""
Created with IntelliJ PyCharm.
@Time: 2022/3/20 9:46
@PROJECT_NAME: chhRL
@File: A2C.py
@Author: chh3213
@Email:
@Description:
Advantage Actor-Critic
    Reference:
        blog:https://blog.csdn.net/weixin_42301220/article/details/123311478
             https://openai.com/blog/baselines-acktr-a2c/
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
        prob = F.softmax(self.fc3(x))
        return prob


class Q(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64):
        super(Q, self).__init__()
        self.q = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                               nn.ReLU(),
                               nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        q = self.q(x)
        return q


class V(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64):
        super(V, self).__init__()
        self.v = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                               nn.ReLU(),
                               nn.Linear(hidden_dim, 1))

    def forward(self, x):
        v = self.v(x)
        return v


class Critic(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.v = V(in_dim, out_dim, hidden_dim=64)
        self.q = Q(in_dim, out_dim, hidden_dim=64)

    def forward(self, x):
        v = self.v(x)
        q = self.q(x)
        advantage = q - v.repeat(2)
        return advantage


class A2C:
    def __init__(self, state_dim, act_dim, args):
        self.a_log_prob = None
        self.s_dim = state_dim
        self.a_dim = act_dim
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(self.s_dim, self.a_dim, self.args.hidden_dim).to(self.device)
        self.critic = Critic(self.s_dim, self.a_dim, self.args.hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.args.lr_actor)
        self.v_optimizer = torch.optim.Adam(self.critic.v.parameters(), self.args.lr_critic)
        self.q_optimizer = torch.optim.Adam(self.critic.q.parameters(), self.args.lr_critic)

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
        # r = torch.tensor(r, device=self.device, dtype=torch.float32)
        v = self.critic.v(s)
        q = self.critic.q(s)[a]
        advantage = q - v
        v_ = self.critic.v(s_)
        v_target = r + self.args.gamma * v_
        td_error = v_target.detach() - v

        # q_target = r+(1-done)*self.args.gamma*torch.max(self.critic.q(s_))
        if not done:
            q_target = torch.max(self.critic.q(s_)) * self.args.gamma + r
            q_loss = (q - q_target.detach()) ** 2
        else:
            q_target = r
            q_loss = (q - q_target) ** 2
        # loss = nn.MSELoss()
        # q_loss = loss(q,q_target)
        v_loss = td_error ** 2

        a_prob = self.actor(s)[a]
        actor_loss = -(torch.log(a_prob) * advantage.detach())

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def save(self, path):
        torch.save(self.actor.state_dict(), path + 'actor.pt')
        torch.save(self.critic.state_dict(), path + 'critic.pt')

    def load(self, path):
        self.actor.load_state_dict(torch.load(path + 'actor.pt'))
        self.critic.load_state_dict(torch.load(path + 'critic.pt'))
