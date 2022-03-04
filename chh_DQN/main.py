# --*--coding: utf-8 --*--
"""
Created with IntelliJ PyCharm.
@Time: 2022/3/3 15:07
@PROJECT_NAME: chhRL
@File: main.py
@Author: chh3213
@Email:
@Description:
"""
import argparse
import os
import sys
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch


def parseSetting():
    """
    参数设置
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default="CartPole-v0",
                        help="environment name：CartPole-v0/CartPole-v1...")
    parser.add_argument('--algo', default="Double_DQN",
                        help="algo choice：Nature_DQN/Naive_DQN/Double_DQN/Dueling_DQN")
    """------------超参数设置------------------"""
    parser.add_argument('--lr', default=0.001, type=float, help="learning rate")
    parser.add_argument('--epsilon', default=0.1, type=float, help="epsilon greedy")
    parser.add_argument('--gamma', default=0.99, type=float, help="discount factor")
    parser.add_argument('--batch_size', default=64, type=int, help="batch size")
    parser.add_argument('--buffer_size', default=100000, type=int, help="size of replay buffer")
    parser.add_argument('--tau', default=0.1, type=float, help="soft update param")
    parser.add_argument('--hidden_dim', default=64, type=int, help="number of hidden layer")
    parser.add_argument('--seed', default=1, type=int, help="seed of random")
    parser.add_argument('--episodes', default=200, type=int, help="")
    parser.add_argument('--steps', default=1000, type=int, help="")
    parser.add_argument('--target_update_frequency', default=1, type=int, help="update frequency of target network ")

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
    print('环境：%s'.format(args.env_name))
    model_dir = args.saveModel_dir + args.algo + '/' + args.env_name
    data_dir = args.saveData_dir + args.algo + '/' + args.env_name
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    rewards = []  # 记录奖励
    ma_rewards = []  # 记录滑动平均奖励
    for i_ep in range(args.episodes):
        ep_reward = 0  # 记录每个回合的奖励
        state = env.reset()  # 重置环境,即开始新的回合
        while True:
            # env.render()
            # 根据算法选择一个动作
            action = agent.choose_action(state)
            # print(action)
            # 与环境进行一次动作交互
            next_state, reward, done, _ = env.step(action)
            # print(done)
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # 更新状态
            state = next_state
            ep_reward += reward
            agent.update()
            if done:
                break

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
    # lines1 = np.loadtxt(data_dir + '/rewards.txt',
    #                     comments="#", delimiter="\n", unpack=False)
    plt.plot(rewards)
    plt.savefig(data_dir + '/rewards.png')
    plt.show()
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
            time.sleep(0.05)
            action = agent.predict(state)  # 根据算法选择一个动作
            next_state, reward, done, _ = env.step(action)  # 与环境进行一个交互
            agent.replay_buffer.push(state, action, reward, next_state, done)
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
    # env = env.unwrapped  # 不加这个的话openai的环境每个episode都有固定的步数限制，加了之后会取消掉
    # 随机数种子
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if type(env.action_space) != gym.spaces.discrete.Discrete:
        state_dim = env.observation_space.shape[0]  # 状态维度
        action_dim = env.action_space.shape[0]  # 动作维度
        print("连续环境的下的DQN暂不实现，只实现离散动作的环境。")
        sys.exit()
    else:
        state_dim = env.observation_space.shape[0]  # 状态维度
        action_dim = env.action_space.n  # 动作维度
        if args.algo == 'DQN_PER':
            # 优先经验回放参数
            hyperparameters = {
                    "batch_size": args.batch_size,
                    "buffer_size": args.buffer_size,
                    "epsilon_decay_rate_denominator": 200,
                    "alpha_prioritised_replay": 0.6,
                    "beta_prioritised_replay": 0.4,
                    "incremental_td_error": 1e-8,
            }
            from DQN_PER import DQN
            agent = DQN(state_dim, action_dim, args, hyperparameters)
        else:
            if args.algo == 'Nature_DQN':
                from Nature_DQN import DQN
            elif args.algo == 'Naive_DQN':
                from Naive_DQN import DQN
            elif args.algo == 'Double_DQN':
                from Double_DQN import DQN
            elif args.algo == 'Dueling_DQN':
                from Dueling_DQN import DQN
            agent = DQN(state_dim, action_dim, args=args)

    print("=====================")
    print("env: ", args.env_name)
    print("algo: ", args.algo)
    print('state dim: ', state_dim)
    print('action dim: ', action_dim)
    print("==========================")
    if args.train:
        if args.restore:
            agent.load(args.saveModel_dir + args.algo + '/' + args.env_name + '/')
        train(args, env, agent)
    else:
        agent.load(args.saveModel_dir + args.algo + '/' + args.env_name + '/')

        test(args, env, agent)


if __name__ == '__main__':
    args = parseSetting()
    main(args)
