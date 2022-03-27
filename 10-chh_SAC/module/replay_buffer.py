#!/usr/bin/python3
# --*--coding:utf-8--*--
"""
==========chhRL===============
@File: replay_buffer.py
@Time: 2022/3/3 下午1:48
@Author: chh3213
@Description:
    replay buffer 实现
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
        batch = random.sample(self.buffer, batch_size)  # 随机采出小批量转移
        state, action, reward, next_state, done = zip(*batch)  # 解包
        return state, action, reward, next_state, done

    def __len__(self):
        """
        返回当前存储的量
        """
        return len(self.buffer)
