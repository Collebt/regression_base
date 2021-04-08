from numpy.core.arrayprint import _leading_trailing
from numpy.linalg.linalg import solve
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas

import numpy as np
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, number, length, cls=None):
        super().__init__()
        self.length = length
        self.number = number


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        

        return super().__getitem__(idx)

def create_A_idx(number, length):
    r'''
    生成随机数据集
    输入：
    A_num: A的总数
    number:数据中包含的A的数量
    length：数据集的样本量
    '''
    
    A_idx = np.random.randint(low=0, high=number, size=(length, number))
    A_idx_matrix = np.zeros(length, number)

    for i, item in enumerate(A_idx_matrix):
        for j in range(number):
            item[A_idx[i, j]] += 1
    return A_idx_matrix
    


def create_A(number, type='int'):
    r'''
    仿真生成A的具体数值
    '''
    if type == 'int':
        A = np.random.randint(low=1, high=50, size=(1, number))
    elif type == 'uniform':
        A = np.random.rand(1, number)
    else:
        A = np.random.random(size=(1, number))
    print('A is :{}'.format(A))
    return A


def calculate_D(Q, A_idx_matrix, A):
    r'''
    计算D的数值作为训练样本
    '''

    return Q * (A_idx_matrix @ A)

def A_solver(Q, A_idx, D):
    '''
        使用最小二乘法求解A的值
        输入Q:已知的Q
        A_idx：A的随机组合矩阵
        D：观测值
    '''
    b = D / Q
    AtA_inv = np.linalg.inv(A_idx.T @ A_idx)
    return AtA_inv @ A_idx.T @ b


if __name__ == '__main__':
    number = 24
    length = 100
    Q = 3

    A = create_A(number, type=int)#生成A的数值
    A_idx = create_A_idx(number, length)
    D = calculate_D(Q, A_idx, A)
    print('D is:{}'.format(D))

    solved_A = A_solver(Q, A_idx, D)
    error = np.sum(np.abs(solved_A - A), axis=1)

