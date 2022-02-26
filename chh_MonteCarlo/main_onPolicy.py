# --*--coding: utf-8 --*--
"""
Created with IntelliJ PyCharm.
@Time: 2022/2/25 23:05
@PROJECT_NAME: chhRL
@File: main_onPolicy.py
@Author: chh3213
@Email:
@Description:
"""

from MC_OnPolicy import MonteCarlo
import gym
import argparse
import os
import numpy as np
import time
from envs.racetrack_env import RacetrackEnv

def parseSetting():
    """
    参数设置
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default="CliffWalking-v0", help="discrete environment name:FrozenLake-v0,CliffWalking-v0")
    parser.add_argument('--lr', default=0.1, type=float, help="learning rate")
    parser.add_argument('--gamma', default=0.99, type=float, help="discount factor")
    parser.add_argument('--epsilon', default=0.15, type=float, help="e-greedy factor")
    parser.add_argument('--episodes', default=500, type=int, help="")
    parser.add_argument('--test_episode', default=1, type=int, help="")
    parser.add_argument('--saveData_dir',
                        default="./save/mc_onPolicy/data/",
                        help="directory to store all experiment data")
    parser.add_argument('--saveModel_dir',
                        default='./save/mc_onPolicy/models/',
                        help="where to store/load network weights")
    parser.add_argument('--load_dir', default='./save/mc_onPolicy/models/',
                        help="where to load network weights")
    parser.add_argument('--checkpoint_frequency', default=100, type=int,
                        help="how often to checkpoint")
    parser.add_argument('--train', default=False, action="store_true",
                        help="train begin if true")

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
    if not os.path.exists(args.saveModel_dir+args.env_name):
        os.makedirs(args.saveModel_dir+args.env_name)

    if not os.path.exists(args.saveData_dir+args.env_name):
        os.makedirs(args.saveData_dir+args.env_name)
    rewards = []  # 记录奖励
    ma_rewards = []  # 记录滑动平均奖励
    for i_ep in range(args.episodes):
        ep_reward = 0  # 记录每个回合的奖励
        state = env.reset()  # 重置环境,即开始新的回合
        one_ep_traj = []
        while True:
            # 根据算法选择一个动作
            action = agent.choose_action(state)
            print(action)
            # env.render()
            # 与环境进行一次动作交互
            next_state, reward, done, _ = env.step(action)
            # 收集一个episode的数据
            one_ep_traj.append([state,action,reward])
            # 更新状态
            state = next_state
            ep_reward += reward
            if done:
                break
        # 蒙特卡洛更新
        agent.update(one_ep_traj)
        rewards.append(ep_reward)
        if ma_rewards:
            # 每一次迭代获得的总收获ep_reward,会以0.1的份额加入到running_reward。
            ma_rewards.append(ma_rewards[-1] * 0.9 + ep_reward * 0.1)
        else:
            ma_rewards.append(ep_reward)
        if (i_ep + 1) % args.checkpoint_frequency == 0:
            agent.save(args.saveModel_dir+args.env_name+'/')
        np.savetxt(args.saveData_dir+args.env_name + '/rewards.txt', rewards)
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
            time.sleep(0.1)
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
    print('完成测试！')
    return rewards, ma_rewards


def main(args):
    env = gym.make(args.env_name)
    # 随机数种子
    # env.seed(1)
    state_dim = env.observation_space.n  # 状态维度
    action_dim = env.action_space.n  # 动作维度

    agent = MonteCarlo(state_dim, action_dim, args.gamma, args.epsilon)
    if args.train:
        train(args, env, agent)
    else:
        q_table = agent.load(args.load_dir+args.env_name+'/')
        # 最后打印Q表格，看看Q表格的样子。
        # print("Final Q-Table Values:/n %s" % q_table)
        test(args, env, agent)


if __name__ == '__main__':
    args = parseSetting()
    main(args)