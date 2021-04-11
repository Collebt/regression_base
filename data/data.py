import numpy as np
from numpy import random
import torch
from torch.utils.data import Dataset
from utils.config import cfg


class MyDataset(Dataset):
    def __init__(self, name, sets, Q, number, length, A, A_better, **args):
        super().__init__()
        self.name = name
        self.length = length
        self.number = number
        # self.ds = eval(self.name)(**args)
        self.stes = sets

        self.A = torch.tensor(A)
        self.A_better = torch.tensor(A_better)
        self.A_idx_sum, self.A_idxs = create_A_idx(
            number, length)  # 自主抽样得到矩阵
        self.D = torch.tensor(calculate_D(Q, self.A_idx_sum, np.array(A)))
        self.Q = torch.tensor(Q)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = {}
        A_idxs = self.A_idxs[idx]
        pos_idx = torch.zeros(self.number, len(self.A))
        for i, A_idx in enumerate(A_idxs):  # 构建索引数据
            pos_idx[i, A_idx] = 1

        data['D'] = self.D[idx]
        data['Q'] = self.Q
        data['A_input'] = self.A[A_idxs].reshape(-1, 1)
        data['A_better'] = self.A_better[A_idxs]
        data['A_pos'] = pos_idx
        data['A_nums'] = torch.tensor(self.A_idx_sum[idx]).reshape(-1, 1)

        return data


def create_A_idx(number, length):
    r'''
    生成随机数据集
    输入：
    A_num: A的总数
    number:数据中包含的A的数量
    length：数据集的样本量
    '''

    A_idx = np.random.randint(low=0, high=number, size=(length, number))
    A_idx_matrix = np.zeros((length, number))

    for i, item in enumerate(A_idx_matrix):
        for j in range(number):
            item[A_idx[i, j]] += 1
    return A_idx_matrix, A_idx


def create_A(number, type='int'):
    r'''
    仿真生成A的具体数值
    '''
    if type == 'int':
        A = np.random.randint(low=1, high=50, size=(number, 1))
    elif type == 'uniform':
        A = np.random.rand(number, 1)
    else:
        A = np.random.random(size=(number, 1))
    # display
    print('A is :{}'.format(A.T))
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


def get_dataloader(dataset, shuffle=False):
    return torch.utils.data.DataLoader(
        dataset, batch_size=cfg.BATCH_SIZE, shuffle=shuffle, num_workers=cfg.DATALOADER_NUM
    )


if __name__ == '__main__':
    number = 24
    length = 100
    Q = 3

    A = create_A(number)  # 生成A的数值
    A_idx = create_A_idx(number, length)  # 自主抽样得到矩阵
    D = calculate_D(Q, A_idx, A)

    solved_A = A_solver(Q, A_idx, D)
    error = np.sum(np.abs(solved_A - A), axis=0)  # 测试求解结果和真实A的误差

    # print
    print('D is:{}'.format(D.T))
    print('error=|A-solved_A| is:{}'.format(error))
