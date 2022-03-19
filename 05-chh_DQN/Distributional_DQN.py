#!/usr/bin/python3
# --*--coding:utf-8--*--
"""
==========chhRL===============
@File: Distributional_DQN.py
@Time: 2022/3/4 下午7:31
@Author: chh3213
@Description:
C51 Algorithm
------------------------
Categorical 51 distributional RL algorithm, 51 means the number of atoms. In
this algorithm, instead of estimating actual expected value, value distribution
over a series of continuous sub-intervals (atoms) is considered.
========Above the sun, full of fire!=============
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from module.utils import hard_update, soft_update
from module.replay_buffer import ReplayBuffer


# from module.network import DistributionalNet

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
        self.q_net = DistributionalNet(self.s_dim, self.a_dim, args.hidden_dim, args.num_atoms).to(self.device)
        self.target_q_net = DistributionalNet(self.s_dim, self.a_dim, args.hidden_dim, args.num_atoms).to(self.device)

        hard_update(self.target_q_net, self.q_net)  # 硬更新方式
        # 声明优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), self.lr)
        # prior knowledge of return distribution
        self.num_atoms = args.num_atoms
        self.V_MIN = -5.
        self.V_MAX = 10.
        self.V_RANGE = np.linspace(self.V_MIN, self.V_MAX, args.num_atoms)
        self.V_STEP = ((self.V_MAX - self.V_MIN) / (args.num_atoms - 1))
        self.value_range = torch.tensor(self.V_RANGE, device=self.device, dtype=torch.float32)  # (N_ATOM)

    def _epsilon_greedy(self, epsilon, q_values):
        if 1 - epsilon > np.random.uniform(0, 1):
            # 选择Q(s,a)最大对应的动作
            action = torch.argmax(q_values, dim=1).item()
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
        # print(np.shape(q_values))
        action_value = torch.sum(q_values * self.value_range.view(1, 1, -1), dim=2)
        # print(np.shape(action_value))
        action = self._epsilon_greedy(self.epsilon, action_value)
        return action

    def predict(self, state):
        """
        策略评估时使用，即测试阶段
        :param state:
        :return:
        """
        state = torch.tensor(state, device=self.device, dtype=torch.float32)
        q_values = self.q_net(state)
        action_value = torch.sum(q_values * self.value_range.view(1, 1, -1), dim=2)
        action = torch.argmax(action_value, dim=1).item()
        return action

    def update(self):
        """
        网络更新操作
        参考自：https://github.com/Kchu/DeepRL_PyTorch/blob/master/Distributional_RL/1_C51.py
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

        # 奖励归一化
        reward_batch = (reward_batch - reward_batch.mean()) / (reward_batch.std() + 1e-7)
        """========注意维度统一和数据类型问题=========="""
        q_values = self.q_net(state_batch)  # # (batch_size, N_ACTIONS, N_ATOM)
        mb_size = q_values.size(0)  # 第1维大小：batch_size
        q_values = torch.stack([q_values[i].index_select(0, action_batch[i]) for i in range(mb_size)]).squeeze(1)
        # print(np.shape(q_values))
        '''==========之前的做法============='''
        # q_next_values = torch.max(self.target_q_net(next_state_batch), 1)[0].detach()  # shape:[batch_size]
        # q_next_values = q_next_values.cpu().numpy()
        """========================="""
        # get next state value
        q_next_values = self.target_q_net(next_state_batch).detach()  # (m, N_ACTIONS, N_ATOM)
        # next value mean
        q_next_mean = torch.sum(q_next_values * self.value_range.view(1, 1, -1), dim=2)  # (m, N_ACTIONS)
        best_actions = q_next_mean.argmax(dim=1)  # (m)
        q_next_values = torch.stack([q_next_values[i].index_select(0, best_actions[i]) for i in range(mb_size)]).squeeze(1)
        q_next_values = q_next_values.data.cpu().numpy()  # (m, N_ATOM)

        # categorical projection
        '''
        next_v_range : (z_j) i.e. values of possible return, shape : (m, N_ATOM)
        next_v_pos : relative position when offset of value is V_MIN, shape : (m, N_ATOM)
        '''
        q_target = np.zeros((mb_size, self.num_atoms))  # (m, N_ATOM)
        # we vectorized the computation of support and position
        next_v_range = np.expand_dims(reward_batch.cpu().numpy(), 1) + self.gamma * np.expand_dims(
            (1. - done_batch.cpu().numpy()), 1) \
                       * np.expand_dims(self.value_range.cpu().numpy(), 0)
        # clip for categorical distribution
        next_v_range = np.clip(next_v_range, self.V_MIN, self.V_MAX)
        # calc relative position of possible value
        next_v_pos = (next_v_range - self.V_MIN) / self.V_STEP
        # get lower/upper bound of relative position
        lb = np.floor(next_v_pos).astype(int)
        ub = np.ceil(next_v_pos).astype(int)
        # we didn't vectorize the computation of target assignment.
        for i in range(mb_size):
            for j in range(self.num_atoms):
                # calc prob mass of relative position weighted with distance
                q_target[i, lb[i, j]] += (q_next_values * (ub - next_v_pos))[i, j]
                q_target[i, ub[i, j]] += (q_next_values * (next_v_pos - lb))[i, j]
        q_target = torch.tensor(q_target, device=self.device, dtype=torch.float32)
        """======================="""
        """原来的loss计算"""
        # loss = nn.MSELoss()
        # q_loss = loss(q_values, q_target)
        """-------------------"""
        loss = q_target*(-torch.log(q_values+1e-8))
        q_loss = torch.mean(loss)
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()

        if self.update_cnt % self.target_update_frequency == 0:
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
