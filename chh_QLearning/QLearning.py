# --*--coding: utf-8 --*--
"""
@Time: 2022/2/23 21:14
@PROJECT_NAME: chhRL
@File: QLearning
@Author: chh3213
@Email :
@Description:
    Q-Table learning algorithm.
    Non deep learning - TD Learning, Off-Policy, e-Greedy Exploration
    Q(S, A) <- Q(S, A) + alpha * (R + lambda * max Q(newS, ) - Q(S, A))
    See David Silver RL Tutorial Lecture 5 - Q-Learning for more details.
    reference to https://github.com/datawhalechina/easy-rl/tree/master/codes/QLearning
    EN: https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0#.5m3361vlw
"""
import numpy as np
import torch
import dill


class Qlearning(object):
    def __init__(self, s_dim, a_dim, lr, gamma, epsilon) -> None:
        """
        初始化
        :param s_dim: 状态维度
        :param a_dim: 动作维度
        :param lr: 学习率
        :param gamma: 折扣率
        :param epsilon: e-greedy 的贪心率
        """
        super().__init__()
        self.state_dim = s_dim
        self.action_dim = a_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        # 建立Q表格，并初始化为全0数组。形状为：[状态空间，动作空间]
        self.Q_table = np.zeros([self.state_dim, self.action_dim])

    def choose_action(self, state):
        """
        使用e-greedy 进行agent动作的选取
        :param state: agent当前状态
        :return:返回选择后的动作
        """
        # e-greedy
        # 有 1-epislon的概率进行最大化选取，epislon的概率随机取得
        if 1 - self.epsilon > np.random.uniform(0, 1):
            # 选择Q(s,a)最大对应的动作
            action = np.argmax(self.Q_table[state, :])
        else:
            # 随机选择动作
            action = np.random.choice(self.action_dim)
        return action

    def predict(self, state):
        """
        动作的估计--测试时使用
        :param state: 当前状态
        :return:
        """
        action = np.argmax(self.Q_table[state, :])
        return action

    def update(self, state, action, reward, next_state, done):
        """
        q table 的更新
        :param state: 当前状态
        :param action: 当前动作
        :param reward: 当前所获得的奖励
        :param next_state: 下一状态
        :param done: 是否终止
        :return:
        """
        Q_predict = self.Q_table[state, action]
        # 如果为true，则为终止状态
        if done:
            Q_target = reward
        else:
            # 计算TD target
            Q_target = reward + self.gamma * np.max(self.Q_table[next_state, :])
        # 更新q table
        self.Q_table[state, action] = self.Q_table[state, action] + self.lr * (Q_target - Q_predict)

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
