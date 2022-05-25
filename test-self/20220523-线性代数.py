#!/usr/bin/env python
# encoding: utf-8
'''
# Author: Cole Chan
# Contact: chensqi@nuist.edu.cn
# Software: Vscode
# File: 20220523-线性代数.py
# Time: 2022/05/23 11:08:14
# Desc:
'''

# here put the import lib

# %%
import torch

x = torch.tensor([3.0])
y = torch.tensor([2.0])
torch.arange(24).reshape(2, 3, 4)

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()
A * B
# 两个矩阵的按元素乘法为 哈达玛积（Hadamard product）
a = 2
X = torch.arange(24).reshape(2, 3, 4)

A = torch.arange(20*2).reshape(2, 5, 4)
sumA = A.sum(axis=1, keepdims=True)  # 广播机制
A/sumA

# 点积是相同位置按元素乘积求和
y = torch.ones(4, dtype=torch.float32)
x = torch.arange(4, dtype=torch.float32)
torch.dot(x, y)
# 矩阵向量积
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
torch.mv(A, x)
B = torch.ones(4,3)
torch.mm(A, B)
#范数
torch.norm(A)