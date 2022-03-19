# --*--coding: utf-8 --*--
"""
Created with IntelliJ PyCharm.
@Time: 2022/2/28 17:57
@PROJECT_NAME: chhRL
@File: main.py
@Author: chh3213
@Email:
@Description:
"""

import gym
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import time

import torch


def parseSetting():
    """
    参数设置
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default="CartPole-v0",
                        help="environment name：CartPole-v0/Pendulum-v0")
    parser.add_argument('--lr', default=0.001, type=float, help="learning rate")
    parser.add_argument('--gamma', default=0.99, type=float, help="discount factor")
    parser.add_argument('--episodes', default=5000, type=int, help="")
    parser.add_argument('--steps', default=1000, type=int, help="")
    parser.add_argument('--test_episode', default=10, type=int, help="")
    parser.add_argument('--saveData_dir',
                        default="./save/data/",
                        help="directory to store all experiment data")
    parser.add_argument('--saveModel_dir',
                        default='./save/models/',
                        help="where to store/load network weights")
    parser.add_argument('--load_dir', default='./save/models/',
                        help="where to load network weights")
    parser.add_argument('--checkpoint_frequency', default=10, type=int,
                        help="how often to checkpoint")
    parser.add_argument('--train', default=False, action="store_true",
                        help="train begin if true")
    parser.add_argument('--restore', default=False, action="store_true",
                        help="restore training begin if true")
    parser.add_argument('--baseline', default=True, action="store_true",
                        help="use REINFORCE with baseline if true")

    args = parser.parse_args()
    return args


def train(args, env, agent):
    """
    训练代码
    :param args: 参数
    :param env: 环境
    :param agent: RL算法生成的智能体
    :return:
    """
    print('train begin!!!!')
    print('环境：%s'.format(args.env_name))
    model_dir = args.saveModel_dir + args.env_name + '/baseline-' + str(args.baseline)
    data_dir = args.saveData_dir + args.env_name + '/baseline-' + str(args.baseline)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    rewards = []  # 记录奖励
    ma_rewards = []  # 记录滑动平均奖励
    for i_ep in range(args.episodes):
        ep_reward = 0  # 记录每个回合的奖励
        state = env.reset()  # 重置环境,即开始新的回合
        rewards_episode = []
        states_episode = []
        while True:
            # env.render()
            # 根据算法选择一个动作
            action = agent.choose_action(state)
            # print(action)
            # 与环境进行一次动作交互
            next_state, reward, done, _ = env.step(action)
            # 更新状态
            state = next_state
            ep_reward += reward
            rewards_episode.append(reward)
            states_episode.append(state)
            if done:
                break
        if i_ep > 0:
            agent.update(rewards_episode, states_episode)
        rewards.append(ep_reward)
        if ma_rewards:
            # 每一次迭代获得的总收获ep_reward,会以0.1的份额加入到running_reward。
            ma_rewards.append(ma_rewards[-1] * 0.9 + ep_reward * 0.1)
        else:
            ma_rewards.append(ep_reward)
        if (i_ep + 1) % args.checkpoint_frequency == 0:
            agent.save(model_dir + '/')
            print('save success')
        np.savetxt(data_dir + '/rewards.txt', rewards)
        print("回合数：{}/{}，奖励{:.1f}".format(i_ep + 1, args.episodes, ep_reward))
    print('完成训练！')
    return rewards, ma_rewards


def test(args, env, agent):
    print('开始测试！')
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 滑动平均的奖励
    for i_ep in range(args.test_episode):
        ep_reward = 0  # 记录每个episode的reward
        state = env.reset()  # 重置环境, 重新开一局（即开始新的一个回合）
        while True:
            env.render()
            action = agent.choose_action(state)  # 根据算法选择一个动作
            next_state, reward, done, _ = env.step(action)  # 与环境进行一个交互
            state = next_state  # 更新状态
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1] * 0.9 + ep_reward * 0.1)
        else:
            ma_rewards.append(ep_reward)
        print(f"回合数：{i_ep + 1}/{args.test_episode}, 奖励：{ep_reward:.1f}")
    env.close()
    print('完成测试！')
    return rewards, ma_rewards


def main(args):
    env = gym.make(args.env_name)
    env.seed(1)
    # torch.manual_seed(1)
    # np.random.seed(1)
    if type(env.action_space) != gym.spaces.discrete.Discrete:
        from REINFORCE_continuous import REINFORCEPolicy
        state_dim = env.observation_space.shape[0]  # 状态维度
        action_dim = env.action_space.shape[0]  # 动作维度
    else:
        if args.baseline:
            from REINFORCE_with_Baseline import REINFORCEPolicy
        else:
            from REINFORCE_discrete import REINFORCEPolicy
        state_dim = env.observation_space.shape[0]  # 状态维度
        action_dim = env.action_space.n  # 动作维度

    agent = REINFORCEPolicy(state_dim, action_dim, hidden_dim=64, args=args)
    if args.train:
        # 随机数种子
        if args.restore:
            agent.load(args.saveModel_dir + args.env_name + '/baseline-' + str(args.baseline) + '/')
        train(args, env, agent)
    else:
        agent.load(args.saveModel_dir + args.env_name + '/baseline-' + str(args.baseline) + '/')

        # lines1 = np.loadtxt('./save/reinforce_discrete/data/CartPole-v0/rewards.txt',
        #                     comments="#", delimiter="\n", unpack=False)
        # plt.plot(lines1)
        # plt.show()
        test(args, env, agent)


if __name__ == '__main__':
    args = parseSetting()
    main(args)
