# --*--coding: utf-8 --*--
"""
Created with IntelliJ PyCharm.
@Time: 2022/3/20 14:29
@PROJECT_NAME: chhRL
@File: main_td3.py
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
from module.utils import OUNoise
from tensorboardX import SummaryWriter


def parseSetting():
    """
    参数设置
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default="Pendulum-v0",
                        help="environment name：Pendulum-v0")
    parser.add_argument('--algo', default="td3",
                        help="algo choice：td3")
    # 学习参数设置
    parser.add_argument('--lr_actor', default=0.0001, type=float, help="learning rate of actor network")
    parser.add_argument('--lr_critic', default=0.001, type=float, help="learning rate of critic network")
    parser.add_argument('--gamma', default=0.99, type=float, help="discount factor")
    parser.add_argument('--hidden_dim', default=256, type=int, help="nums of hidden layer")
    parser.add_argument('--episodes', default=100, type=int, help="")
    parser.add_argument('--steps', default=1000, type=int, help="")
    parser.add_argument('--seed', default=1, type=int, help="seed of random")
    parser.add_argument('--buffer_size', default=100000, type=int, help="size of replay buffer")
    parser.add_argument('--batch_size', default=64, type=int, help="batch size")

    # td3参数设置
    parser.add_argument('--update_freq', default=2, type=int, help="policy update frequency")
    parser.add_argument('--noise_std', default=0.2, type=float, help="policy update frequency")
    parser.add_argument('--noise_clip', default=0.5, type=float, help="policy update frequency")


    # 数据保存
    parser.add_argument('--test_episode', default=10, type=int, help="")
    parser.add_argument('--saveData_dir',
                        default="./save/data/",
                        help="directory to store all experiment data")
    parser.add_argument('--saveModel_dir',
                        default='./save/models/',
                        help="where to store/load network weights")
    parser.add_argument('--load_dir', default='./save/models/',
                        help="where to load network weights")
    parser.add_argument('--checkpoint_frequency', default=50, type=int,
                        help="how often to checkpoint")
    parser.add_argument('--train', default=False, action="store_true",
                        help="train begin if true")
    parser.add_argument('--restore', default=False, action="store_true",
                        help="restore training begin if true")

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
    print('环境：{}'.format(args.env_name))
    model_dir = args.saveModel_dir + args.env_name + "/" + args.algo
    data_dir = args.saveData_dir + args.env_name + "/" + args.algo
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    writer = SummaryWriter(data_dir + '/logs/')
    ou_noise = OUNoise(env.action_space)  # 动作噪声
    rewards = []  # 记录奖励
    ma_rewards = []  # 记录滑动平均奖励
    for i_ep in range(args.episodes):
        ep_reward = 0  # 记录每个回合的奖励
        state = env.reset()  # 重置环境,即开始新的回合
        i_step = 0
        while True:
            i_step += 1
            # env.render()
            # 根据算法选择一个动作
            action = agent.choose_action(state)
            action = ou_noise.get_action(action, i_step)
            # print(action)
            # 与环境进行一次动作交互
            next_state, reward, done, _ = env.step(action)
            # 更新状态
            ep_reward += reward
            writer.add_scalar("reward", ep_reward, global_step=i_ep)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            if done:
                break
            actor_loss, critic_1_loss,critic_2_loss = agent.update()
            writer.add_scalar("Loss/actor_loss", actor_loss, global_step=i_ep)
            writer.add_scalar("Loss/critic_1_loss", critic_1_loss, global_step=i_ep)
            writer.add_scalar("Loss/critic_2_loss", critic_2_loss, global_step=i_ep)
            state = next_state

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
    # env = env.unwrapped
    env.seed(args.seed)
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    print("agent action_dim:", a_dim)
    print("agent state_dim:", s_dim)
    max_action = float(env.action_space.high[0])
    from td3 import TD3
    agent = TD3(state_dim=s_dim, action_dim=a_dim, max_action=max_action, args=args)

    if args.train:
        # 随机数种子
        if args.restore:
            agent.load(args.saveModel_dir + args.env_name + "/" + args.algo + "/")
        train(args, env, agent)
    else:
        # env = env.unwrapped
        agent.load(args.saveModel_dir + args.env_name + "/" + args.algo + "/")

        # lines1 = np.loadtxt('./save/reinforce_discrete/data/CartPole-v0/rewards.txt',
        #                     comments="#", delimiter="\n", unpack=False)
        # plt.plot(lines1)
        # plt.show()
        test(args, env, agent)


if __name__ == '__main__':
    args = parseSetting()
    main(args)
