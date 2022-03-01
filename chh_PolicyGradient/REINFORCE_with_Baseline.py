# --*--coding: utf-8 --*--
"""
Created with IntelliJ PyCharm.
@Time: 2022/3/1 19:16
@PROJECT_NAME: chhRL
@File: REINFORCE_with_Baseline.py
@Author: chh3213
@Email:
@Description:
    Baseline: Instead of measuring the absolute goodness of an action
    we want to know how much better than "average" it is to take an action given a state.
    E.g. some states are naturally bad and always give negative reward.
     This is called the advantage and is defined as Q(s, a) - V(s).
     We use that for our policy update, e.g. g_t - V(s) for REINFORCE.
 Reference to :https://blog.csdn.net/weixin_42301220/article/details/123112474
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.distributions import Categorical
from torch.autograd import Variable


class PolicyNet(nn.Module):
    """
    定义一个简单的全连接层网络，用于得到策略
    """

    def __init__(self, input_dim, output_dim, hidden_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = F.softmax(self.fc3(x))
        return output


class VNet(nn.Module):
    """
    定义一个简单的全连接层网络,用于表示状态价值函数
    """

    def __init__(self, input_dim, output_dim, hidden_dim):
        super(VNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = F.softmax(self.fc3(x))
        return output


class REINFORCEPolicy:
    def __init__(self, state_dim, action_dim, hidden_dim, args):
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = args.lr
        self.gamma = args.gamma
        self.log_probs = []
        # 一个浮点数
        self.eps = np.finfo(np.float32).eps.item()
        # 如果有gpu则使用gpu
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.policy_net = PolicyNet(self.s_dim, self.a_dim, hidden_dim).to(self.device)
        # 价值函数网络
        self.V_net = VNet(self.s_dim, 1, hidden_dim).to(self.device)
        print(self.device)
        self.optimizerP = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.optimizerV = torch.optim.Adam(self.V_net.parameters(), lr=self.lr)

    def choose_action(self, state):
        """
        根据当前状态选择动作
        :param state:
        :return:
        """
        # print(state)
        state = torch.from_numpy(state).float()
        state = state.to(self.device)
        state = Variable(state)
        probs = self.policy_net(state)

        # 分类分布
        m = Categorical(probs)
        # 采样一个action
        action = m.sample()
        # m.log_prob(action) 就是 logp
        self.log_probs.append(m.log_prob(action))
        return action.item()

    def predict(self):
        pass

    def update(self, rewards_episode, states_episode):
        """
        更新网络
        :param rewards_episode: 一个episode的奖励，计算总回报用
        :param states_episode: 一个episode的状态，计算状态价值函数用
        （其实可以直接在choose_action函数中计算，这样就不用再传一次，写法可见REINFORCE_discrete.py）
        :return:
        """
        # 总收益
        R = 0
        rewards = []
        loss_p_sum = 0
        loss_v_sum = 0
        for r in rewards_episode[::-1]:
            # r为每一步的奖励
            if r == 0:
                R = 0
            else:
                R = r + self.gamma * R
            rewards.insert(0, R)
        # 对回报进行normalize
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps)

        for log_prob, reward, state in zip(self.log_probs, rewards, states_episode):
            state = torch.from_numpy(state).float()
            v = self.V_net(Variable(state).to(self.device))
            # 优势函数
            delta = reward - v
            # 计算loss
            loss_p = -(log_prob * delta)
            loss_v = -delta
            loss_v_sum += loss_v
            loss_p_sum += loss_p
            # 每步计算梯度
            # 反向传播计算梯度
        # 一个episode的所有loss相加再计算梯度
        # 更新权重
        self.optimizerP.zero_grad()
        # 不加retain_graph=True，计算两个梯度的时候会报错
        loss_p_sum.backward(retain_graph=True)
        self.optimizerP.step()

        self.optimizerV.zero_grad()
        loss_v_sum.backward(retain_graph=True)
        self.optimizerV.step()
        self.log_probs = []

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path + 'model_p.pt')
        torch.save(self.V_net.state_dict(), path + 'model_v.pt')

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path + 'model_p.pt'))
        self.V_net.load_state_dict(torch.load(path + 'model_v.pt'))
