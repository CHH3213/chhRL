# --*--coding: utf-8 --*--
"""
Created with IntelliJ PyCharm.
@Time: 2022/2/26 9:29
@PROJECT_NAME: chhRL
@File: testcode.py
@Author: chh3213
@Email:
@Description:
测试python语法代码，与强化学习无关
"""
from collections import defaultdict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
#         x = F.max_pool2d(F.relu(self.conv2(x)), 2)
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


# net = Net()
# # print(net)
#
# params = list(net.parameters())
# # print(len(params))
# # print(params[0].size())
# input = torch.randn(1,3,32,32)
# # print(np.shape(input))
# out = net(input)
# # print(out)
# # net.zero_grad()
# # out.backward(torch.randn(1, 10))
# criterion = nn.MSELoss()
# target = torch.randn(10)  # 随机真值
# # print(target)
# target = target.view(1, -1)  # 变成行向量
#
# output = net(input)  # 用随机输入计算输出
#
# loss = criterion(output, target)  # 计算损失
# print(loss)
# net.zero_grad()
#
# print('conv1.bias.grad before backward')
# print(net.conv1.bias.grad)
# print(net.fc2.bias.grad)
# loss.backward()

# print('conv1.bias.grad after backward')
# print(net.conv1.bias.grad)
# opt = optim.Adam(net.parameters(),lr=0.01)
# opt.zero_grad()
# output = net(input)
# loss = criterion(output, target)
# loss.backward()
# # 更新权重
# opt.step()


# rewards=[]
# rewards.insert(0,5)
# rewards.insert(0,4)
# print(rewards)


# a = torch.tensor([[[3, 5], [2, 3], [6, 7]]])
# print(a)
# print(np.shape(a))
# a = torch.unsqueeze(a, 0)
# print(np.shape(a))
# # print(a)
# a = torch.squeeze(a)
# print(np.shape(a))
# # print(a)
#
#
# a = torch.tensor([[3, 5], [2, 3], [6, 7]])
# b = torch.max(a, 0)[0]
# c = torch.max(a, 1)[0]
# print('a:',a)
# print('b:',b)
# print('c:',c)

'测试bool的done转成float'
# done = torch.tensor(False,dtype=torch.int64)
# done = np.array([0,0,0,1,0,1])
# print(1-done)


# '测试torch.gather'
# a = torch.Tensor([[1,2],[3,4]])
# print(a)
# a = torch.gather(a,1,index=torch.tensor([[0],[1]]))
# print(a)

# import random
# finish = random.randint(2,10)
# print(finish)

# 假设是时间步T1
T1 = torch.tensor([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
# 假设是时间步T2
T2 = torch.tensor([[10, 20, 30],
                   [40, 50, 60],
                   [70, 80, 90]])

print(torch.stack((T1, T2)).shape)


# print(torch.stack((T1, T2), dim=1).shape)
# print(torch.stack((T1, T2), dim=2).shape)


class Net1:
    def __init__(self, input_dim=16, output_dim=1, hidden1_dim=64, hidden2_dim=32):
        # 随机产生一些随机特征
        features = torch.randn(1, input_dim)
        # 构建权重
        w1 = torch.randn((input_dim, hidden1_dim), requires_grad=True)
        w2 = torch.randn((hidden1_dim, hidden1_dim), requires_grad=True)
        w3 = torch.randn((hidden1_dim, hidden2_dim), requires_grad=True)
        w4 = torch.randn((hidden2_dim, output_dim), requires_grad=True)
        # 构建偏置
        b1 = torch.randn((hidden1_dim), requires_grad=True)
        b2 = torch.randn((hidden1_dim), requires_grad=True)
        b3 = torch.randn((hidden2_dim), requires_grad=True)
        b4 = torch.randn((output_dim), requires_grad=True)

        # 构造隐藏层和输出层
        h1 = F.relu((features @ w1) + b1)
        h2 = F.relu((h1 @ w2) + b2)
        h3 = F.relu((h2 @ w3) + b3)
        out = F.sigmoid((h3 @ w4) + b4)


class Net2(nn.Module):
    def __init__(self, input_dim=16, output_dim=1, hidden1_dim=64, hidden2_dim=32):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden1_dim)
        self.fc2 = nn.Linear(hidden1_dim, hidden1_dim)
        self.fc3 = nn.Linear(hidden1_dim, hidden2_dim)
        self.fc4 = nn.Linear(hidden2_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        out = F.sigmoid(self.fc4(x))
        return out


# model = Net2()
# print(model)
# # 检查权重示例
# print(model.fc2.weight)
#
# input_dim=16
# output_dim=1
# hidden1_dim=64
# hidden2_dim=32
# model = nn.Sequential(
#     nn.Linear(input_dim, hidden1_dim),
#     nn.ReLU(),
#     nn.Linear(hidden1_dim, hidden1_dim),
#     nn.ReLU(),
#     nn.Linear(hidden1_dim, hidden2_dim),
#     nn.ReLU(),
#     nn.Linear(hidden2_dim, output_dim),
#     nn.Sigmoid()
# )
# print(model)
# print(model[0].weight)
#
#
# from collections import OrderedDict
#
#
# model = nn.Sequential(OrderedDict([
#     ('fc1', nn.Linear(input_dim, hidden1_dim)),
#     ('relu1', nn.ReLU()),
#     ('fc2', nn.Linear(hidden1_dim, hidden1_dim)),
#     ('relu2', nn.ReLU()),
#     ('fc3', nn.Linear(hidden1_dim, hidden2_dim)),
#     ('relu3', nn.ReLU()),
#     ('fc4', nn.Linear(hidden2_dim, output_dim)),
#     ('sigmoid', nn.Sigmoid())
# ])
# )
# print(model)
# print(model.fc2.weight)
#
#
#
#
# class Net4(nn.Module):
#     def __init__(self, input_dim=16, output_dim=1, hidden1_dim=64, hidden2_dim=32):
#         super(Net4, self).__init__()
#         self.layers = nn.Sequential(OrderedDict([
#             ('fc1', nn.Linear(input_dim, hidden1_dim)),
#             ('relu1', nn.ReLU()),
#             ('fc2', nn.Linear(hidden1_dim, hidden1_dim)),
#             ('relu2', nn.ReLU()),
#             ('fc3', nn.Linear(hidden1_dim, hidden2_dim)),
#             ('relu3', nn.ReLU()),
#             ('fc4', nn.Linear(hidden2_dim, output_dim)),
#         ])
#         )
#
#     def forward(self, x):
#         out = F.sigmoid(self.layers(x))
#         return out
# model4 = Net4()
# print(model4)


import torch
print(torch.cuda.is_available())
