# --*--coding: utf-8 --*--
"""
Created with IntelliJ PyCharm.
@Time: 2022/2/27 22:28
@PROJECT_NAME: chhRL
@File: value_iteration.py
@Author: chh3213
@Email:
@Description:
"""
import numpy as np
from envs.gridworld import GridworldEnv

class ValueIteration:
    def __init__(self, env, gamma, theta):
        self.env = env
        self.state_dim = env.nS
        self.action_dim = env.nA
        # print(env.nS)  # 16
        # print(env.nA)  # 4
        # 折扣因子
        self.gamma = gamma
        # s small positive number determining the accuracy of estimation
        # 详见policy iteration的伪代码
        self.theta = theta
    def random_policy(self):
        """
        随机策略
        :return:
        """
        return np.ones([self.state_dim, self.action_dim]) / self.action_dim

    def value_iter(self):
        V = np.zeros(self.state_dim)
        # 第一步，计算最优价值函数
        while True:
            delta = 0
            for s in range(self.state_dim):
                A = np.zeros(self.action_dim)
                # 遍历动作空间中所有的action
                for a in range(self.action_dim):
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        A[a] += prob * (reward + self.gamma * V[next_state])
                # 计算V，具体见伪代码更新V的式子
                max_v = np.max(A)
                # 计算delta
                delta = max(delta, np.abs(max_v - V[s]))
                V[s] = max_v
            if delta < self.theta:
                break
        # 第二步，通过最优价值函数确定策略
        policy = np.zeros([self.state_dim, self.action_dim])
        for s in range(self.state_dim):
            A = np.zeros(self.action_dim)
            # 遍历所有的action
            for a in range(self.action_dim):
                for prob, next_state, reward, done in self.env.P[s][a]:
                    A[a] += prob * (reward + self.gamma * V[next_state])
            best_action = np.argmax(A)
            policy[s,best_action] = 1.0
        return V,policy


if __name__ == '__main__':
    env = GridworldEnv()
    valueIteration = ValueIteration(env, gamma=1.0, theta=0.00001)
    V,policy = valueIteration.value_iter()
    print('======policy==========')
    print(policy)
    print('=========value function=========')
    print(V)

