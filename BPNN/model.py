import torch
import torch.nn as nn
import torch.nn.functional as F


def MLP(channels: list, do_bn=True):  # 多层感知器的生成
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class Net(nn.Module):
    r'''
    网络骨架框架
    '''

    def __init__(self, A_nums, out_features, optim_layers, predict_layers):
        super(Net, self).__init__()
        self.num_inputs = A_nums
        self.num_outputs = out_features
        self.optimA = OptimeA(A_nums, optim_layers)  # 实例化优化模块
        self.predict = PredNN(A_nums, predict_layers)  # 实例化预测模块
        self.output = nn.Linear(A_nums, 1)

    def forward(self, x, position, A_nums):
        # 输入Aij： x.shape=(1,24,1)  Aij的位置索引idx.shape=(1,24,24)
        x_pred = self.optimA(x, position)
        # x_pred.shape=(1,1,24) , idx.shape=(1,24,24)
        pred = self.predict(x_pred, position)  # resnet框架
        pred = self.output(x_pred + pred)  # 预测Q的值
        return pred, x_pred


class OptimeA(nn.Module):
    '''
    优化Aij模块
    '''

    def __init__(self, A_nums, layers=[16, 14, 8, 4]):
        super(OptimeA, self).__init__()
        self.optimA = MLP([A_nums+1] + layers + [1])

    def forward(self, x, poisition):

        inputs = [x.transpose(1, 2), poisition.transpose(1, 2)]
        # 输入为第一维方向拼接
        return x.transpose(1, 2) + self.optimA(torch.cat(inputs, dim=1))


class PredNN(nn.Module):
    '''
    预测Q模块
    '''

    def __init__(self, A_nums, layers=[16, 14, 8, 4]):
        super().__init__()
        self.encoder = MLP([A_nums+1] + layers + [1])
        # nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, x_pred, poisition):
        '''
        x_pred.shape=(1 * 1 * 24) , poisition.shape=(1 * 24 * 24)
        '''
        inputs = [x_pred, poisition.transpose(1, 2)]
        # 转换数据类型为float32
        return self.encoder(torch.cat(inputs, dim=1))


class MyReLU(torch.autograd.Function):
    '''
    自定义可以储存梯度的激活函数
    '''
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backword(input)  # 储存输入参数
        return input.clamp(min=0)  # ReLU激活函数

    @staticmethod
    def backwrad(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input
