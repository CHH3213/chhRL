# --*--coding: utf-8 --*--
"""
Created with IntelliJ PyCharm.
@Time: 2022/3/21 14:49
@PROJECT_NAME: chhRL
@File: ddpg.py
@Author: chh3213
@Email:
@Description:
    DDPG suffers from problems like overestimate of Q-values and sensitivity to hyper-parameters.
    The implementation of TD3 includes 6 networks: 2 Q-net, 2 target Q-net, 1 policy net, 1 target policy net
    Actor policy in TD3 is deterministic, with Gaussian exploration noise.
"""

import torch
import copy
import numpy as np
import torch.nn.functional as F
from module.replay_buffer import ReplayBuffer
from module.network import Actor, Critic
from module.utils import soft_update


class TD3:
    def __init__(self, state_dim, action_dim, max_action, args):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, action_dim, self.args.hidden_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.args.lr_actor)

        self.critic_1 = Critic(state_dim, action_dim, self.args.hidden_dim).to(self.device)
        self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), self.args.lr_critic)

        self.critic_2 = Critic(state_dim, action_dim, self.args.hidden_dim).to(self.device)
        self.critic_2_target = copy.deepcopy(self.critic_2)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), self.args.lr_critic)

        self.replay_buffer = ReplayBuffer(self.args.buffer_size, self.args.seed)

        self.total_iter = 0  # 延迟更新使用

    def choose_action(self, state):
        state = torch.tensor(state, device=self.device, dtype=torch.float32)
        action = self.actor(state)
        return action.detach().cpu().numpy().flatten()

    def predict(self):
        pass

    def update(self):
        self.total_iter += 1
        if len(self.replay_buffer) < self.args.batch_size:  # 当 memory 中不满足一个批量时，不更新策略
            return 0, 0,0
        # 从经验回放中(replay memory)中随机采样一个批量的转移(transition)
        state, action, reward, next_state, done = self.replay_buffer.sample(self.args.batch_size)
        # 转换成tensor
        state = torch.tensor(state, device=self.device, dtype=torch.float32)
        action = torch.tensor(action, device=self.device, dtype=torch.float32)
        next_state = torch.tensor(next_state, device=self.device, dtype=torch.float32)
        reward = torch.tensor(reward, device=self.device, dtype=torch.float32)  # shape([batch_size])
        done = torch.tensor(done, device=self.device, dtype=torch.float32)  # shape([batch_size])

        """trick 1: Select action according to policy and add clipped noise"""
        next_action = self.actor_target(next_state)
        noise = torch.normal(torch.zeros(next_action.size()), self.args.noise_std).to(self.device)
        noise = torch.clamp(noise, -self.args.noise_clip, self.args.noise_clip)
        next_action = (next_action + noise).clamp(-self.max_action,self.max_action)

        """trick 2: double Q, 计算target Q"""
        target_Q1 = self.critic_1_target(next_state, next_action).squeeze()
        target_Q2 = self.critic_2_target(next_state, next_action).squeeze()
        target_Q = torch.min(target_Q1, target_Q2)
        # 注意reward，done，target_Q的维度要统一
        target_Q = reward + ((1 - done) * self.args.gamma * target_Q)

        # 计算当前Q
        Q1 = self.critic_1(state, action).squeeze()
        Q2 = self.critic_2(state, action).squeeze()
        # 计算q loss,注意target_Q和Q维度要统一
        critic_1_loss = F.mse_loss(target_Q, Q1)
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward(retain_graph=True)
        self.critic_1_optimizer.step()
        #
        critic_2_loss = F.mse_loss(target_Q, Q2)
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward(retain_graph=True)
        self.critic_2_optimizer.step()

        # critic_loss = F.mse_loss(target_Q, Q1) + F.mse_loss(target_Q, Q2)
        # self.critic_1_optimizer.zero_grad()
        # self.critic_2_optimizer.zero_grad()
        # critic_loss.backward()
        # self.critic_1_optimizer.step()
        # self.critic_2_optimizer.step()

        """trick3:Delayed policy updates"""
        # 计算p loss
        actor_loss = self.critic_1(state, self.actor(state))
        actor_loss = -actor_loss.mean()
        if self.total_iter % self.args.update_freq == 0:
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 更新目标网络参数，软更新方式
            soft_update(self.actor_target, self.actor, tau=0.001)
            soft_update(self.critic_1_target, self.critic_1, tau=0.001)
            soft_update(self.critic_2_target, self.critic_2, tau=0.001)
        return actor_loss.item(), critic_1_loss.item(), critic_2_loss.item()

    def save(self, path):
        torch.save(self.critic_1.state_dict(), path + "critic_1.pt")
        torch.save(self.critic_1.state_dict(), path + "critic_2.pt")
        torch.save(self.critic_1_optimizer.state_dict(), path + "critic_1_optimizer.pt")
        torch.save(self.critic_1_optimizer.state_dict(), path + "critic_2_optimizer.pt")
        torch.save(self.actor.state_dict(), path + "actor.pt")
        torch.save(self.actor_optimizer.state_dict(), path + "actor_optimizer.pt")

    def load(self, path):
        self.actor.load_state_dict(torch.load(path + "actor.pt"))
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer.load_state_dict(torch.load(path + "actor_optimizer.pt"))
        self.critic_1.load_state_dict(torch.load(path + "critic_1.pt"))
        self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_1_optimizer.load_state_dict(torch.load(path + "critic_1_optimizer.pt"))
        self.critic_2.load_state_dict(torch.load(path + "critic_2.pt"))
        self.critic_2_target = copy.deepcopy(self.critic_2)
        self.critic_2_optimizer.load_state_dict(torch.load(path + "critic_2_optimizer.pt"))
