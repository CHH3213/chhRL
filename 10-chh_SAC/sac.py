# --*--coding: utf-8 --*--
"""
Created with IntelliJ PyCharm.
@Time: 2022/3/27 17:38
@PROJECT_NAME: chhRL
@File: sac.py
@Author: chh3213
@Email:
@Description:
"""
import copy

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal

from module.replay_buffer import ReplayBuffer
from module.network import Stochastic_Actor, Critic, V_Net
from module.utils import soft_update


class SAC:
    def __init__(self, state_dim, action_dim, args):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.actor = Stochastic_Actor(state_dim, action_dim, self.args.hidden_dim).to(self.device)

        self.V = V_Net(state_dim, self.args.hidden_dim).to(self.device)
        self.target_V = copy.deepcopy(self.V)

        self.Q = Critic(state_dim, action_dim, self.args.hidden_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.args.lr_actor)
        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), self.args.lr_q)
        self.V_optimizer = torch.optim.Adam(self.V.parameters(), self.args.lr_v)

        self.replay_buffer = ReplayBuffer(self.args.buffer_size, self.args.seed)

    def choose_action(self, state):
        """训练时与环境交互使用"""
        state = torch.tensor(state, device=self.device, dtype=torch.float32)
        mu, log_sigma = self.actor(state)
        dist = Normal(mu, torch.exp(log_sigma))
        action = dist.sample()
        action = torch.tanh(action)
        return action.detach().cpu().numpy().flatten()

    def predict(self):
        pass

    def update(self):
        if len(self.replay_buffer) < self.args.batch_size:  # 当 memory 中不满足一个批量时，不更新策略
            return
        # 从经验回放中(replay memory)中随机采样一个批量的转移(transition)
        state, action, reward, next_state, done = self.replay_buffer.sample(self.args.batch_size)
        # 转换成tensor
        state = torch.tensor(state, device=self.device, dtype=torch.float32)
        action = torch.tensor(action, device=self.device, dtype=torch.float32)
        next_state = torch.tensor(next_state, device=self.device, dtype=torch.float32)
        reward = torch.tensor(reward, device=self.device, dtype=torch.float32)  # shape([batch_size])
        done = torch.tensor(done, device=self.device, dtype=torch.float32)  # shape([batch_size])
        # 升高一维，跟后面q的维度相匹配，计算loss才不会有问题
        reward = reward.unsqueeze(1)
        done = done.unsqueeze(1)
        self.v_learn(state)
        self.q_learn(state, action, reward, next_state, done)
        self.actor_learn(state)
        soft_update(self.target_V, self.V)

    def q_learn(self, state, action, reward, next_state, done):
        """
        q网络更新
        :param state:
        :param action:
        :return:
        """
        q_value = self.Q(state, action)
        target_q = reward + (1 - done) * self.args.gamma * self.target_V(next_state)
        loss = nn.MSELoss()
        q_loss = loss(q_value, target_q.detach())
        q_loss = q_loss.mean()

        self.Q_optimizer.zero_grad()
        q_loss.backward()
        self.Q_optimizer.step()

    def v_learn(self, state):
        """
        v 网络更新
        :param state:
        :return:
        """
        mu, log_sigma = self.actor(state)
        dist = Normal(mu, torch.exp(log_sigma))
        action = dist.sample()
        log_prob = dist.log_prob(action)
        action = torch.tanh(action)

        q_value = self.Q(state, action)
        v_value = self.V(state)

        target_v = q_value - log_prob

        loss = nn.MSELoss()
        v_loss = loss(v_value, target_v.detach())
        v_loss = v_loss.mean()
        self.V_optimizer.zero_grad()
        v_loss.backward()
        self.V_optimizer.step()

    def actor_learn(self, state):
        mu, log_sigma = self.actor(state)
        dist = Normal(mu, torch.exp(log_sigma))
        action = dist.sample()
        log_prob = dist.log_prob(action)
        action = torch.tanh(action)

        q_value = self.Q(state, action)
        v_value = self.V(state)

        target_log_prob = q_value - v_value
        policy_loss = log_prob * (log_prob - target_log_prob).detach()
        policy_loss = policy_loss.mean()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

    def save(self, path):
        torch.save(self.Q.state_dict(), path + "q.pt")
        torch.save(self.actor.state_dict(), path + "actor.pt")
        torch.save(self.V.state_dict(), path + "v.pt")
        torch.save(self.actor_optimizer.state_dict(),path+'actor_optimizer.pt')
        torch.save(self.V_optimizer.state_dict(),path+'V_optimizer.pt')
        torch.save(self.Q_optimizer.state_dict(),path+'Q_optimizer.pt')

    def load(self, path):
        self.actor.load_state_dict(torch.load(path + "actor.pt"))
        self.Q.load_state_dict(torch.load(path + "q.pt"))
        self.V.load_state_dict(torch.load(path + "v.pt"))
        self.target_V = copy.deepcopy(self.V)

        self.Q_optimizer.load_state_dict(torch.load(path + "Q_optimizer.pt"))
        self.V_optimizer.load_state_dict(torch.load(path + "V_optimizer.pt"))
        self.actor_optimizer.load_state_dict(torch.load(path + "actor_optimizer.pt"))
