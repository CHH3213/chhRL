#!/usr/bin/python3
# --*--coding:utf-8--*--
"""
==========chhRL===============
@File: ppo_replay_buffer
@Time: 2022/3/25 下午2:55
@Author: chh3213
@Description:
========Above the sun, full of fire!=============
"""
import random


class ReplayBuffer:
    def __init__(self, capacity, seed=0):
        random.seed(seed)  # 随机数种子
        self.capacity = capacity  # 经验回放的容量
        self.buffer = []  # 缓冲区
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """
        缓冲区是一个队列，容量超出时去掉开始存入的转移(transition)
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.buffer:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
        return s, a, r, s_prime, done_mask, prob_a

    def __len__(self):
        """
        返回当前存储的量
        """
        return len(self.buffer)
