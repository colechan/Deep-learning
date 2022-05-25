#!/usr/bin/env python
# encoding: utf-8
'''
# Author: Cole Chan
# Contact: chensqi@nuist.edu.cn
# Software: Vscode
# File: 20220523-矩阵计算.py
# Time: 2022/05/23 15:03:29
# Desc:
'''

# here put the import lib

# %%
import torch 
x = torch.arange(4,dtype=torch.float32)
x.requires_grad_(True)
x.grad
y = 2 * torch.dot(x,x)
y.backward()
x.grad == 4 * x

# x.grad.zero_()
y = x.sum()
y.backward()
x.grad


x.grad.zero_()
y = x * x
y.sum().backward()
x.grad

x.grad.zero_()
y = x * x
u = y.detach()
z = u * x
z.sum().backward()
x.grad