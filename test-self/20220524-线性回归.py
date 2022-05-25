#!/usr/bin/env python
# encoding: utf-8
'''
# Author: Cole Chan
# Contact: chensqi@nuist.edu.cn
# Software: Vscode
# File: 20220524-线性回归.py
# Time: 2022/05/24 10:06:02
# Desc:
'''

# here put the import lib
# %%

import random
import torch
from d2l import torch as d2l

def synthetic_data(w,b,num_examples):
    '''生成 y = Wx + b + 噪声'''
    X = torch.normal(0,1,(num_examples,len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0,0.01,y.shape)
    return X,y.reshape((-1,1)),y

true_w = torch.tensor([2,-3.4])
true_b = 4.2
features,labels,y = synthetic_data(true_w,true_b,1000)
d2l.set_figsize()
d2l.plt.scatter(features[:,0].detach().numpy(),labels.detach().numpy(),1)
# %%
# 将大样本转为随机的小批量样本
def data_iter(batch_size,features,labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices) # 随机打乱
    for i in range(0,num_examples,batch_size):
        batch_indices = torch.tensor(indices[i:min(i+batch_size,num_examples)])
        yield features[batch_indices],labels[batch_indices]
        
batch_size = 10
i = 0
for X,y in data_iter(batch_size,features,labels):
    print(X,'\n',y)
    if i == 0 : break

# %%
w = torch.normal(0,0.01,size=(2,1),requires_grad=True) #需要计算梯度
b = torch.zeros(1,requires_grad=True)
# 定义模型
def linreg(X, y, b):
    return torch.matmul(X, w) + b
# 定义损失函数
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape))**2/2
# 可能是行向量，也可能是列向量
# 定义优化算法
def sgd(params, lr, batch_size):
    '''小批量随机梯度下降
    params 给定所有的参数，包括w和b
    lr 给定学习率
    '''
    with torch.no_grad(): # 不需要计算梯度：更新的时候不需要
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_() # 手动设置为0，下一次计算梯度不会和上一次相关
# 训练过程
lr = 10  # 超参数，学习率
num_epochs = 30
net = linreg # 模型统一定义
loss = squared_loss # 均方损失

for epoch in range(num_epochs): #第一层对所有数据扫一遍
    for X, y in data_iter(batch_size,features,labels):
        l = loss(net(X, w, b),y)
        # l 的形状是(batch_size,1),不是一个标量。所以求梯度的时候需要累加然后再求
        l.sum().backward()
        sgd([w,b], lr, batch_size) # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features,w,b),labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
# %%        
#简洁实现
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)        

def load_array(data_arrays, batch_size, is_train = True):
    '''构造pytorch迭代器'''
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train) # shuffle代表是否要打乱顺序

batch_size = 10
data_iter = load_array((features, labels), batch_size)
next(iter(data_iter))
# %%
from torch import nn

net = nn.Sequential(nn.Linear(2, 1))
# 线性回归就是简单的单层神经网络
net[0].weight.data.normal_(0,0.01)
# normal_ 是代表使用正态分布替换掉data的数值
net[0].bias.data.fill_(0)

loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr = 0.03)
# 两个参数，（所有参数），学习率
# 训练模块
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')