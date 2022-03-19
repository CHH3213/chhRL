#!/usr/bin/python3
# --*--coding:utf-8--*--
"""
==========chhRL===============
@File: Noisy_DQN.py
@Time: 2022/3/4 下午2:45
@Author: chh3213
@Description:
========Above the sun, full of fire!=============
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from module.utils import hard_update, soft_update
from module.replay_buffer import ReplayBuffer
from module.network import NoisyNet



class DQN:
    def __init__(self, state_dim, action_dim, args):
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = args.lr  # 学习率
        self.gamma = args.gamma  # 奖励的折扣因子
        self.epsilon = args.epsilon  # 贪心策略
        self.tau = args.tau  # 软更新参数
        # 如果有gpu则使用gpu
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 经验池设置
        self.batch_size = args.batch_size
        self.replay_buffer = ReplayBuffer(args.buffer_size, args.seed)
        # q网络更新次数
        self.update_cnt = 0
        self.target_update_frequency = args.target_update_frequency  # 目标网络更新频率
        # 网络初始化
        self.q_net = NoisyNet(self.s_dim, self.a_dim, args.hidden_dim).to(self.device)
        self.target_q_net = NoisyNet(self.s_dim, self.a_dim, args.hidden_dim).to(self.device)

        hard_update(self.target_q_net, self.q_net)  # 硬更新方式
        # 声明优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), self.lr)

    def _epsilon_greedy(self, epsilon, q_values):
        if 1 - epsilon > np.random.uniform(0, 1):
            # 选择Q(s,a)最大对应的动作
            action = torch.argmax(q_values).item()
        else:
            # 随机选择动作
            action = np.random.choice(self.a_dim)
        return action

    def choose_action(self, state):
        """
        训练阶段用来选取动作
        :param state:
        :return:
        """
        state = torch.tensor(state, device=self.device, dtype=torch.float32)
        # print(state)
        q_values = self.q_net(state)
        # print(q_values)
        action = self._epsilon_greedy(self.epsilon, q_values)
        return action

    def predict(self, state):
        """
        策略评估时使用，即测试阶段
        :param state:
        :return:
        """
        state = torch.tensor(state, device=self.device, dtype=torch.float32)
        q_values = self.q_net(state)
        action = torch.argmax(q_values).item()
        return action

    def update(self):
        """
        网络更新操作
        :return:
        """
        # 当经验池中不满足一个批量时，不更新策略
        if len(self.replay_buffer) < self.batch_size:
            return
        self.update_cnt += 1
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(
            self.batch_size)
        # 转为tensor,维度均为[batch_size]
        state_batch = torch.tensor(state_batch, device=self.device, dtype=torch.float32)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float32)
        # 注意action是int类型的离散变量
        action_batch = torch.tensor(action_batch, device=self.device, dtype=torch.int64)
        next_state_batch = torch.tensor(next_state_batch, device=self.device, dtype=torch.float32)
        done_batch = torch.tensor(done_batch, device=self.device, dtype=torch.int64)

        # reward_batch = (reward_batch - reward_batch.mean()) / (reward_batch.std() + 1e-7)
        """========注意维度统一问题=========="""
        # print(np.shape(self.q_net(state_batch))) # [batch_size,2]
        # 将action_batch升高1维，维度变为[batch_size,1]
        action_batch = action_batch.unsqueeze(1)
        # 计算Q(s,a)。 torch.gather函数:沿给定轴dim，将输入索引张量index指定位置的值进行聚合。
        q_values = torch.gather(input=self.q_net(state_batch), dim=1, index=action_batch)  # shape:[batch_size,1]
        """Nature DQN 计算方式"""
        # 求出max Q^(s',)，与伪代码一致
        q_next_values = torch.max(self.target_q_net(next_state_batch), 1)[0].detach()  # shape:[batch_size]
        q_target = reward_batch + self.gamma * q_next_values * (1 - done_batch)
        """==================="""
        loss = nn.MSELoss()
        # print(np.shape(q_target))
        # q_values维度为[batch_size,1]，减少1维与q_target保持一致，即shape变为[batch_size],这样才可对应计算loss
        q_loss = loss(q_values.squeeze(), q_target)
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()

        if self.update_cnt%self.target_update_frequency==0:
            # 硬更新
            # hard_update(self.target_q_net, self.q_net)
            # 软更新
            soft_update(self.target_q_net, self.q_net, self.tau)

    def save(self, path):
        torch.save(self.q_net.state_dict(), path + 'q_net.pt')
        torch.save(self.target_q_net.state_dict(), path + 'target_q_net.pt')

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path + 'q_net.pt'))
        self.target_q_net.load_state_dict(torch.load(path + 'target_q_net.pt'))

