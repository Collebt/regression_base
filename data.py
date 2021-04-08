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

def create_data(A_num, number, length):
    '''
    生成随机数据集
    输入：
    A_num: A的总数
    number:数据中包含的A的数量
    length：数据集的样本量
    '''
    
    A_idx = np.random.randint(low=0, high=A_num, size=(length, number))
    



def create_A(number, type='int'):
    if type == 'int':
        A = np.random.randint(number)
    elif type == 'uniform':
        A = np.random.rand(number)
    else:
        A = np.random.random(number)
    print('A is :{}'.format(A))
    return A


