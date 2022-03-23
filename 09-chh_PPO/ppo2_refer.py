# --*--coding: utf-8 --*--
"""
Created with IntelliJ PyCharm.
@Time: 2022/3/22 21:48
@PROJECT_NAME: chhRL
@File: ppo2_refer.py
@Author: chh3213
@Email:
@Description:
"""
import gym
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class ActorNet(nn.Module):
    def __init__(self, n_states, bound):
        super(ActorNet, self).__init__()
        self.n_states = n_states
        self.bound = bound

        self.layer = nn.Sequential(
            nn.Linear(self.n_states, 128),
            nn.ReLU()
        )

        self.mu_out = nn.Linear(128, 1)
        self.sigma_out = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.layer(x))
        mu = self.bound * torch.tanh(self.mu_out(x))
        sigma = F.softplus(self.sigma_out(x))
        return mu, sigma


class CriticNet(nn.Module):
    def __init__(self, n_states):
        super(CriticNet, self).__init__()
        self.n_states = n_states

        self.layer = nn.Sequential(
            nn.Linear(self.n_states, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        v = self.layer(x)
        return v


class PPO(nn.Module):
    def __init__(self, n_states, n_actions, bound, args):
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.bound = bound
        self.lr = args.lr
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.a_update_steps = args.a_update_steps
        self.c_update_steps = args.c_update_steps

        self._build()

    def _build(self):
        self.actor_model = ActorNet(n_states, bound)
        self.actor_old_model = ActorNet(n_states, bound)
        self.actor_optim = torch.optim.Adam(self.actor_model.parameters(), lr=self.lr)

        self.critic_model = CriticNet(n_states)
        self.critic_optim = torch.optim.Adam(self.critic_model.parameters(), lr=self.lr)

    def choose_action(self, s):
        s = torch.FloatTensor(s)
        # print(np.shape(s))
        mu, sigma = self.actor_model(s)
        dist = torch.distributions.Normal(mu, sigma)
        action = dist.sample()
        return np.clip(action, -self.bound, self.bound)

    def discount_reward(self, rewards, s_):
        s_ = torch.FloatTensor(s_)
        target = self.critic_model(s_).detach()  # torch.Size([1])
        target_list = []
        for r in rewards[::-1]:
            target = r + self.gamma * target
            target_list.append(target)
        target_list.reverse()
        target_list = torch.cat(target_list)  # torch.Size([batch])

        return target_list

    def actor_learn(self, states, actions, advantage):
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions).reshape(-1, 1)
        # print(np.shape(states))
        # print(np.shape(actions))
        mu, sigma = self.actor_model(states)
        pi = torch.distributions.Normal(mu, sigma)

        old_mu, old_sigma = self.actor_old_model(states)
        old_pi = torch.distributions.Normal(old_mu, old_sigma)

        ratio = torch.exp(pi.log_prob(actions) - old_pi.log_prob(actions))
        surr = ratio * advantage.reshape(-1, 1)  # torch.Size([batch, 1])
        loss = -torch.mean(
            torch.min(surr, torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage.reshape(-1, 1)))

        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

    def critic_learn(self, states, targets):
        states = torch.FloatTensor(states)
        v = self.critic_model(states).reshape(1, -1).squeeze(0)

        loss_func = nn.MSELoss()
        loss = loss_func(v, targets)

        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

    def cal_adv(self, states, targets):
        states = torch.FloatTensor(states)
        v = self.critic_model(states)  # torch.Size([batch, 1])
        advantage = targets - v.reshape(1, -1).squeeze(0)
        return advantage.detach()  # torch.Size([batch])

    def update(self, states, actions, targets):
        self.actor_old_model.load_state_dict(self.actor_model.state_dict())  # 首先更新旧模型
        advantage = self.cal_adv(states, targets)

        for i in range(self.a_update_steps):  # 更新多次
            self.actor_learn(states, actions, advantage)

        for i in range(self.c_update_steps):  # 更新多次
            self.critic_learn(states, targets)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_episodes', type=int, default=600)
    parser.add_argument('--len_episode', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--c_update_steps', type=int, default=10)
    parser.add_argument('--a_update_steps', type=int, default=10)
    args = parser.parse_args()

    env = gym.make('Pendulum-v0')
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    bound = env.action_space.high[0]

    agent = PPO(n_states, n_actions, bound, args)

    all_ep_r = []
    for episode in range(args.n_episodes):
        ep_r = 0
        s = env.reset()
        states, actions, rewards = [], [], []
        for t in range(args.len_episode):
            a = agent.choose_action(s)
            s_, r, done, _ = env.step(a)
            ep_r += r
            states.append(s)
            actions.append(a)
            rewards.append((r + 8) / 8)  # 参考了网上的做法

            s = s_

            if (t + 1) % args.batch == 0 or t == args.len_episode - 1:  # N步更新
                states = np.array(states)
                actions = np.array(actions)
                rewards = np.array(rewards)

                targets = agent.discount_reward(rewards, s_)  # 奖励回溯
                agent.update(states, actions, targets)  # 进行actor和critic网络的更新
                states, actions, rewards = [], [], []

        print('Episode {:03d} | Reward:{:.03f}'.format(episode, ep_r))

        if episode == 0:
            all_ep_r.append(ep_r)
        else:
            all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r * 0.1)  # 平滑

    plt.plot(np.arange(len(all_ep_r)), all_ep_r)
    plt.show()

