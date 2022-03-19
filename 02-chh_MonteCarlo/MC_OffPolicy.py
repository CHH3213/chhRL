# --*--coding: utf-8 --*--
"""
Created with IntelliJ PyCharm.
@Time: 2022/2/26 9:40
@PROJECT_NAME: chhRL
@File: MC_OffPolicy.py
@Author: chh3213
@Email:
@Description:
    Monte Carlo Control using Epsilon-Greedy policies.
    Off-Policy First-Visit MC Control algorithm,
    Detailed steps refer to the pseudo code.
    CN:https://blog.csdn.net/weixin_42301220/article/details/123088029
"""
import numpy as np
import torch
import dill
from collections import defaultdict


class MonteCarlo:
    def __init__(self, s_dim, a_dim, gamma):
        self.state_dim = s_dim
        self.action_dim = a_dim
        self.gamma = gamma
        # The final action-value function.
        # A dictionary that maps state -> action values
        self.Q_table = defaultdict(lambda: np.zeros(self.action_dim))
        # 初始化C，伪代码中的符号
        # The cumulative denominator of the weighted importance sampling formula
        # (across all episodes)
        self.C = defaultdict(lambda: np.zeros(self.action_dim))
        # 总回报，初始化为空列表
        self.Returns = defaultdict(list)

        self.behavior_policy = self.random_policy()
        self.target_policy = self.greedy_policy()
    def random_policy(self):
        """
        随机选择动作，做behavior policy
        """
        def policy_fn(observation):
            A = np.ones(self.action_dim, dtype=float) / self.action_dim
            return A

        return policy_fn

    def choose_action(self, state):
        """
        随机选择动作，做behavior policy
        :param state:
        :return:
        """
        # 随机选择动作

        a_probs = self.behavior_policy(state)
        action = np.random.choice(np.arange(len(a_probs)), p=a_probs)
        # print(action,'+++')
        return action

    def greedy_policy(self):
        """
        使用greedy 基于Q值进行agent动作的选取,在这里作为target policy
        :param state: agent当前状态
        :return:返回选择后的动作
        """
        def policy_fn(state):
            A = np.zeros_like(self.Q_table[state], dtype=float)
            best_action = np.argmax(self.Q_table[state])
            A[best_action] = 1.0
            print(A)
            return A

        return policy_fn


    def predict(self, state):
        """
        动作的估计--测试时使用， 为target policy
        :param state: 当前状态
        :return:
        """
        a_probs = self.target_policy(state)
        action = np.random.choice(np.arange(len(a_probs)), p=a_probs)
        return action

    def update(self, one_ep_traj):
        """
        agent的更新
        :param one_ep_traj: 一个episode的轨迹：每个轨迹点包括：（state,action,reward）
        :return:
        """
        # 总回报G
        G = 0
        # 重要性采样权重
        W = 1.0
        for t in range(len(one_ep_traj) - 2, -1, -1):
            s_t, a_t = one_ep_traj[t][0], one_ep_traj[t][1]
            r_t_1 = one_ep_traj[t + 1][2]
            # 更新总回报
            G = self.gamma * G + r_t_1
            # 更新C
            self.C[s_t][a_t] += W
            self.Q_table[s_t][a_t] += (W / self.C[s_t][a_t]+1e-60) * (G - self.Q_table[s_t][a_t])
            # If the action taken by the behavior policy is not the action
            # taken by the target policy the probability will be 0 and we can break
            if a_t == np.argmax(self.target_policy(s_t)):
                break
            W = W * 1.0 / (self.behavior_policy(s_t)[a_t])
            print('self.Q_table[s_t][a_t]', self.Q_table[s_t][a_t])

    def save(self, path):
        """
        保存模型
        :param path: 保存模型路径
        :return:
        """
        torch.save(obj=self.Q_table,
                   f=path + "model.pkl",
                   pickle_module=dill)
        print(" save model success!!!!")

    def load(self, path):
        """
        加载模型
        :param path: 加载模型路径
        :return:
        """
        self.Q_table = torch.load(f=path + "model.pkl", pickle_module=dill)
        print("模型加载成功")
        return self.Q_table
