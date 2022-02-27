# --*--coding: utf-8 --*--
"""
Created with IntelliJ PyCharm.
@Time: 2022/2/27 21:24
@PROJECT_NAME: chhRL
@File: policy_iteration.py
@Author: chh3213
@Email:
@Description: Policy iteration algorithm
reference to：
            https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Policy%20Iteration%20Solution.ipynb
            http://incompleteideas.net/book/RLbook2018.pdf
            https://blog.csdn.net/weixin_42301220/article/details/123069770
"""
import numpy as np
from envs.gridworld import GridworldEnv


class PolicyIteration:
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

    def policy_eval(self, policy):
        """
        伪代码第一阶段：策略评估，对于给定的策略计算V
        :param policy: 这边给定随机策略
        :return:
        """
        # 价值函数
        V = np.zeros(self.state_dim)
        while True:
            delta = 0
            for s in range(self.state_dim):
                v = 0
                # 对当前的策略pi
                for a, action_prob in enumerate(policy[s]):
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        # 伪代码步骤2的V(s)的更新公式
                        v += action_prob * prob * (reward + self.gamma * V[next_state])
                delta = max(delta, np.abs(v - V[s]))
                V[s] = v
            if delta < self.theta:
                break
        return V

    def policy_improvement(self, policy):
        """
        伪代码第二阶段：策略评估，对于给定的策略计算V
        价值 函数过后，可以进一步推算出它的 Q 函数，然后直接在 Q 函数上面取极大化，进一步改进它的策略。
        伪代码policy_evaluation与policy_improvement不停迭代
        :param policy: 即伪代码中的pai
        :return:
        """
        while True:
            # 评估当前的策略
            V = self.policy_eval(policy)
            # 当策略有任何改变时，设为false
            policy_stable = True
            for s in range(self.state_dim):
                # 当下采取的最好动作，注意，这里policy的列索引才表示action
                old_action = np.argmax(policy[s])
                A = np.zeros(self.action_dim)
                # 遍历所有的action
                for a in range(self.action_dim):
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        A[a] += prob * (reward + self.gamma * V[next_state])
                best_action = np.argmax(A)
                policy[s] = np.eye(self.action_dim)[best_action]
                if old_action != best_action:
                    policy_stable = False
            # print(A)
            if policy_stable:
                break
        return V, policy


if __name__ == '__main__':
    env = GridworldEnv()
    policyIteration = PolicyIteration(env, gamma=1.0, theta=0.00001)
    rand_policy = policyIteration.random_policy()
    V, policy = policyIteration.policy_improvement(rand_policy)
    print(policy)
    print(V)
