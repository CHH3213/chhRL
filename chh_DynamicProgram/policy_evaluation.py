# --*--coding: utf-8 --*--
"""
Created with IntelliJ PyCharm.
@Time: 2022/2/27 17:45
@PROJECT_NAME: chhRL
@File: policy_evaluation.py
@Author: chh3213
@Email:
@Description:
Evaluate a policy given an environment and a full description of the environment's dynamics.
reference to：
            https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Policy%20Evaluation%20Solution.ipynb
            http://incompleteideas.net/book/RLbook2018.pdf
            https://blog.csdn.net/weixin_42301220/article/details/123069770
"""
import numpy as np
from envs.gridworld import GridworldEnv

class Policy_eval:
    def __init__(self, env, gamma, theta):
        self.env = env
        self.state_dim = env.nS
        self.action_dim = env.nA
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

    def policy_eval(self, policy):
        """
        策略评估
        :param policy:
        :return:
        """
        # 价值函数
        V = np.zeros(self.state_dim)
        while True:
            delta = 0
            for s in range(self.state_dim):
                v = 0
                for a, action_prob in enumerate(policy[s]):
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        # 伪代码步骤2的V(s)的更新公式
                        v += action_prob * prob * (reward + self.gamma * V[next_state])
                delta = max(delta, np.abs(v - V[s]))
                V[s]=v
            if delta < self.theta:
                break
        return V


if __name__=='__main__':
    env = GridworldEnv()
    p_eval = Policy_eval(env,gamma=1.0,theta=0.00001)
    rand_policy = p_eval.random_policy()
    V = p_eval.policy_eval(rand_policy)
    print(V)