import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas

import numpy as np


def Q_func(Q_num, x):
    '''
    Q是线性方程，Q(x)=Qx1+Qx2+..+Qxn
    得到矩阵乘法为Qx =  Q^T*x,其中Q为全Q矩阵 A in R^n
    
    '''
    length = len(x)
    Q = np.ones(length) * Q_num
    return np.matmul(Q.T, x)



    




class MyReLU(torch.autograd.Function):
    '''
    自定义可以储存梯度的激活函数
    '''
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backword(input) #储存输入参数
        return input.clamp(min=0) #ReLU激活函数
    @staticmethod
    def backwrad(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input<0]=0
        return grad_input