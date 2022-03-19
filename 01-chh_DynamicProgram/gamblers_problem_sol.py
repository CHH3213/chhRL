# --*--coding: utf-8 --*--
"""
Created with IntelliJ PyCharm.
@Time: 2022/2/27 23:17
@PROJECT_NAME: chhRL
@File: gamblers_problem_sol.py
@Author: chh3213
@Email:
@Description:
A gambler has the opportunity to make bets on the outcomes of a sequence of coin flips.
reference to
    http://incompleteideas.net/book/RLbook2018.pdf
    https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Gamblers%20Problem%20Solution.ipynb

"""
import numpy as np
class Gamblers_sol:
    def __init__(self, p_h, gamma, theta):
        # 折扣因子
        self.gamma = gamma
        # s small positive number determining the accuracy of estimation
        # 详见policy iteration的伪代码
        self.theta = theta
        # 硬币正面朝上的概率
        self.p_h = p_h
    def value_iteration(self):
        V = np.zeros(101)
        rewards = np.zeros(101)
        rewards[100] = 1
        # 第一步，计算最优价值函数
        while True:
            delta = 0
            for s in range(1,100):
                A = np.zeros(101)
                stakes = range(1,min(s,(100-s)+1))
                # 遍历动作空间中所有的action
                for a in stakes:
                    # rewards[s+a], rewards[s-a] are immediate rewards.
                    # V[s+a], V[s-a] are values of the next states.
                    # This is the core of the Bellman equation: The expected value of your action is
                    # the sum of immediate rewards and the value of the next state.
                    A[a]= self.p_h*(rewards[s+a]+V[s+a]*self.gamma)+(1-self.p_h)*(rewards[s-a]+V[s-a]*self.gamma)
                # 计算V，具体见伪代码更新V的式子
                max_v = np.max(A)
                # 计算delta
                delta = max(delta, np.abs(max_v - V[s]))
                V[s] = max_v
            if delta < self.theta:
                break
        # 第二步，通过最优价值函数确定策略
        policy = np.zeros(100)
        for s in range(1,100):
            A = np.zeros(101)
            # 遍历所有的action
            stakes = range(1, min(s, (100 - s) + 1))

            for a in stakes:
                A[a] = self.p_h * (rewards[s + a] + V[s + a] * self.gamma) + (1 - self.p_h) * (
                            rewards[s - a] + V[s - a] * self.gamma)
            best_action = np.argmax(A)
            policy[s] = best_action
        return V,policy


if __name__ == '__main__':
    gamblers_sol = Gamblers_sol(0.25,1.0,0.0001)
    V,policy = gamblers_sol.value_iteration()
    print('======policy==========')
    print(policy)
    print('=========value function=========')
    print(V)