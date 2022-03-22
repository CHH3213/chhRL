# --*--coding: utf-8 --*--
"""
Created with IntelliJ PyCharm.
@Time: 2022/3/22 13:06
@PROJECT_NAME: chhRL
@File: PPO_Penalty.py
@Author: chh3213
@Email:
@Description:ppo1
"""

import copy
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.distributions.categorical import Categorical
from module.utils import soft_update,hard_update
from module.network import Critic, Actor_discrete, Actor_continue, V_Net
from module.replay_buffer import ReplayBuffer


class PPO:
    def __init__(self, state_dim, action_dim, max_action, is_discrete, args):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.max_action = max_action
        self.is_discrete = is_discrete

        # 构建actor网络：
        # actor有两个actor 和 actor_old， actor_old的主要功能是记录行为策略的版本。
        if is_discrete:  # 如果离散
            self.actor = Actor_discrete(state_dim, action_dim, self.args.hidden_dim).to(self.device)
            self.actor_old = Actor_discrete(state_dim, action_dim, self.args.hidden_dim).to(self.device)
        else:
            self.actor = Actor_continue(state_dim, action_dim, max_action, self.args.hidden_dim).to(self.device)
            self.actor_old = Actor_continue(state_dim, action_dim, max_action, self.args.hidden_dim).to(self.device)

        # 构建critic网络：两种critic，这边选择其中一种：输入state，输出V值
        self.critic = V_Net(state_dim, self.args.hidden_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.args.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.args.lr_critic)

        self.replay_buffer = ReplayBuffer(self.args.buffer_size, self.args.seed)

    def choose_action(self, state):
        state = torch.tensor(state, device=self.device, dtype=torch.float32)
        # print(np.shape(state))
        if self.is_discrete:
            a_prob = self.actor(state)
            dist = Categorical(a_prob)
            action = dist.sample()
            return action.item()
        else:
            mu, sigma = self.actor(state)
            # print(mu,sigma)
            dist = Normal(mu, sigma)
            action = dist.sample()
            return np.clip(action.detach().cpu().numpy(), -self.max_action, self.max_action)


    def update(self):
        """
        update actor，critic
        :return:
        """
        hard_update(self.actor_old, self.actor)

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
        # print(np.shape(state))   # shape:[batch_size,num of state]
        # print(np.shape(action))  # shape:[batch_size,num of action]
        # print(np.shape(next_state))  # shape:[batch_size,num of state]
        # print(np.shape(reward))  # shape:[batch_size]
        # print(np.shape(done))  # shape:[batch_size]
        """计算target"""
        # 计算critic_value
        critic_value = self.critic(next_state).squeeze()  # shape:[batch_size]
        # 注意reward，done，critic_value的维度要统一
        target = reward + ((1 - done) * self.args.gamma * critic_value)  # shape:[batch_size]
        """计算优势函数"""
        v = self.critic(state).squeeze()  # shape:[batch_size]
        advantage = (target - v).detach()  # shape:[batch_size]
        """actor和critic更新"""
        for _ in range(self.args.actor_update_steps):
            self._actor_learn(state, action, advantage)
        for _ in range(self.args.critic_update_steps):
            v = self.critic(state).squeeze()  # shape:[batch_size]
            self._critic_learn(v, target)


    def _actor_learn(self, state, action, advantage):
        """
        actor 更新
        :param state: state batch
        :param action:  action batch
        :param advantage: 优势batch
        :return:
        """
        if self.is_discrete:
            a_prob = self.actor(state)
            pi = Categorical(a_prob)
            old_a_prob = self.actor_old(state)
            old_pi = Categorical(old_a_prob)
        else:
            mu, sigma = self.actor(state)
            pi = Normal(mu, sigma)
            old_mu, old_sigma = self.actor_old(state)
            old_pi = Normal(old_mu, old_sigma)
        ratio = (torch.exp(pi.log_prob(action) - old_pi.log_prob(action))).squeeze()
        # print(np.shape(ratio))
        weighted_probs = ratio * advantage
        # print(np.shape(advantage))
        # print(np.shape(weighted_probs))
        weighted_clipped_probs = torch.clamp(ratio, 1 - self.args.policy_clip,
                                             1 + self.args.policy_clip) * advantage
        # print(np.shape(weighted_clipped_probs))
        loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

    def _critic_learn(self, v, target):
        """
        critic 更新
        :param v: 价值函数v batch
        :param target: td target batch
        :return:
        """
        loss = nn.MSELoss()
        # print(np.shape(v))
        # print(np.shape(target))
        critic_loss = loss(v, target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def save(self, path):
        torch.save(self.critic.state_dict(), path + "critic.pt")
        torch.save(self.critic_optimizer.state_dict(), path + "critic_optimizer.pt")
        torch.save(self.actor.state_dict(), path + "actor.pt")
        torch.save(self.actor_optimizer.state_dict(), path + "actor_optimizer.pt")
        torch.save(self.actor_old.state_dict(), path + "actor_old.pt")

    def load(self, path):
        self.actor.load_state_dict(torch.load(path + "actor.pt"))
        self.actor_old.load_state_dict(torch.load(path + "actor_old.pt"))
        self.actor_optimizer.load_state_dict(torch.load(path + "actor_optimizer.pt"))
        self.critic.load_state_dict(torch.load(path + "critic.pt"))
        self.critic_optimizer.load_state_dict(torch.load(path + "critic_optimizer.pt"))
