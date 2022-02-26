# --*--coding: utf-8 --*--
"""
Created with IntelliJ PyCharm.
@Time: 2022/2/25 22:31
@PROJECT_NAME: chhRL
@File: MC_OnPolicy
@Author: chh3213
@Email:
@Description:
    Monte Carlo Control using Epsilon-Greedy policies.
    On-Policy First-Visit MC Control algorithm,
    Detailed steps refer to the pseudo code.
    CN:https://blog.csdn.net/weixin_42301220/article/details/123088029
"""

import numpy as np
import torch
import dill
from collections import defaultdict

class MonteCarlo:
    def __init__(self, s_dim, a_dim, gamma, epsilon):
        self.state_dim = s_dim
        self.action_dim = a_dim
        self.gamma = gamma
        self.epsilon = epsilon
        # 建立Q表格，并初始化为全0数组。形状为：[状态空间，动作空间]
        self.Q_table = np.zeros([self.state_dim, self.action_dim])
        # 总回报，初始化为空列表
        self.Returns =  defaultdict(list)

    def epsilon_greedy_policy(self, state):
        """
        e-greedy 贪心策略
        if best_action = 0, then the probability of action is:
        A[epsilon/4, 1-3*epsilon/4, epsilon/4, epsilon/4]
        :param state: agent当前状态
        :return:
        """
        best_action = np.argmax(self.Q_table[state,:])
        A = np.ones(self.action_dim, dtype=np.float32) * self.epsilon / self.action_dim
        A[best_action] += 1 - self.epsilon
        return A

    def choose_action(self, state):
        """
        使用e-greedy 进行agent动作的选取,训练和测试时的动作选择都是用这个
        :param state: agent当前状态
        :return:返回选择后的动作
        """
        a_probs = self.epsilon_greedy_policy(state)
        action = np.random.choice(np.arange(self.action_dim), p=a_probs)
        return action

    def update(self, one_ep_traj):
        """
        agent的更新
        :param one_ep_traj: 一个episode的轨迹：每个轨迹点包括：（state,action,reward）
        :return:
        """
        # 总回报G
        G = 0
        for t in range(len(one_ep_traj)-2,-1,-1):
            s_t,a_t = one_ep_traj[t][0],one_ep_traj[t][1]
            r_t_1 = one_ep_traj[t+1][2]
            G = self.gamma*G+r_t_1
            if not(s_t,a_t) in [(one_ep_traj[i][0],one_ep_traj[i][1]) for i in range(0,t)]:
                self.Returns[(s_t,a_t)].append(G)
                self.Q_table[s_t][a_t] = np.average(self.Returns[(s_t,a_t)])
                print('self.Q_table[s_t][a_t]',self.Q_table[s_t][a_t])
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
