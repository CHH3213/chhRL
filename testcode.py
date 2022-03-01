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
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(-1,16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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




rewards=[]
rewards.insert(0,5)
rewards.insert(0,4)
print(rewards)
