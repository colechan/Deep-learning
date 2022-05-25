#!/usr/bin/env python
# encoding: utf-8
'''
# Author: Cole Chan
# Contact: chensqi@nuist.edu.cn
# Software: Vscode
# File: 20220523-数据预处理.py
# Time: 2022/05/23 10:38:08
# Desc:
'''

# here put the import lib

# %%
import torch

# %%
x = torch.arange(12)
x.shape
x.numel()  # 元素的个数
x.reshape(3, 4)
torch.ones((2, 3, 4))
# list of list
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 5], [4, 5, 6, 8]])

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x-y, x*y, x/y, x**y
torch.exp(x)

# 张量合并
X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
X == Y
X.sum()

# %%
before = id(Y)
Y = Y + X
id(Y) == before

Z = torch.zeros_like(Y)
id(Z)
Z[:] = X + Y
id(Z)
# %%
import pandas as pd
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
# 如果没有安装pandas，只需取消对以下行的注释来安装pandas
# !pip install pandas
import pandas as pd

data = pd.read_csv(data_file)
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
inputs = pd.get_dummies(inputs, dummy_na=True)

import torch

x, y = torch.tensor(inputs.values), torch.tensor(outputs.values)