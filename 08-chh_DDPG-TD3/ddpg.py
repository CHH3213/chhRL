# --*--coding: utf-8 --*--
"""
Created with IntelliJ PyCharm.
@Time: 2022/3/21 14:49
@PROJECT_NAME: chhRL
@File: ddpg.py
@Author: chh3213
@Email:
@Description:
    An algorithm concurrently learns a Q-function and a policy.
    It uses off-policy data and the Bellman equation to learn the Q-function,
    and uses the Q-function to learn the policy.
"""

import torch
import copy
import numpy as np
import torch.nn.functional as F
from module.replay_buffer import ReplayBuffer
from module.network import Actor, Critic
from module.utils import soft_update


class DDPG:
    def __init__(self, state_dim, action_dim, max_action, args):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, action_dim, self.args.hidden_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.args.lr_actor)

        self.critic = Critic(state_dim, action_dim, self.args.hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.args.lr_critic)

        self.replay_buffer = ReplayBuffer(self.args.buffer_size, self.args.seed)

    def choose_action(self, state):
        state = torch.tensor(state, device=self.device, dtype=torch.float32)
        action = self.actor(state)
        return action.detach().cpu().numpy().flatten()

    def predict(self):
        pass

    def update(self):
        if len(self.replay_buffer) < self.args.batch_size:  # 当 memory 中不满足一个批量时，不更新策略
            return 0, 0
        # 从经验回放中(replay memory)中随机采样一个批量的转移(transition)
        state, action, reward, next_state, done = self.replay_buffer.sample(self.args.batch_size)
        # 转换成tensor
        state = torch.tensor(state, device=self.device, dtype=torch.float32)
        action = torch.tensor(action, device=self.device, dtype=torch.float32)
        next_state = torch.tensor(next_state, device=self.device, dtype=torch.float32)
        reward = torch.tensor(reward, device=self.device, dtype=torch.float32)  # shape([64])
        done = torch.tensor(done, device=self.device, dtype=torch.float32)  # shape([64])

        # 计算target Q
        next_action = self.actor_target(next_state)
        target_Q = self.critic_target(next_state, next_action).squeeze()
        # 注意reward，done，target_Q的维度要统一
        target_Q = reward + ((1 - done) * self.args.gamma * target_Q)
        # 计算当前Q
        Q = self.critic(state, action).squeeze()

        # 计算q loss,注意target_Q和Q维度要统一
        critic_loss = F.mse_loss(target_Q, Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 计算p loss
        actor_loss = self.critic(state, self.actor(state))
        actor_loss = -actor_loss.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新目标网络参数，软更新方式
        soft_update(self.critic_target, self.critic, tau=0.001)
        soft_update(self.actor_target, self.actor, tau=0.001)
        # print(actor_loss.item(), critic_loss.item())
        return actor_loss.item(),critic_loss.item()

    def save(self, path):
        torch.save(self.critic.state_dict(), path + "critic.pt")
        torch.save(self.critic_optimizer.state_dict(), path + "critic_optimizer.pt")
        torch.save(self.actor.state_dict(), path + "actor.pt")
        torch.save(self.actor_optimizer.state_dict(), path + "actor_optimizer.pt")

    def load(self, path):
        self.actor.load_state_dict(torch.load(path + "actor.pt"))
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer.load_state_dict(torch.load(path + "actor_optimizer.pt"))
        self.critic.load_state_dict(torch.load(path + "critic.pt"))
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer.load_state_dict(torch.load(path + "critic_optimizer.pt"))
