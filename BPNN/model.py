import torch
import torch.nn as nn
import torch.nn.functional as F


def MLP(channels: list, do_bn=True):
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
    def __init__(self, A_nums, out_features, optim_layers, predict_layers):
        super(Net, self).__init__()
        self.num_inputs = A_nums
        self.num_outputs = out_features
        self.optimA = OptimeA(A_nums, optim_layers)
        self.predict = PredNN(predict_layers)
        self.output = nn.Linear(A_nums, 1)

    def forward(self, x, position, A_nums):
        # x.shape=(1,24,1)  idx.shape=(1,24,24)
        x_pred = self.optimA(x, position)
        # x_pred.shape=(1,1,24) , A_num.shape=(1,24,1),
        pred = self.predict(x_pred, A_nums.transpose(1, 2))  # resnet
        pred = self.output(x_pred + pred)
        return pred, x_pred


class OptimeA(nn.Module):
    def __init__(self, A_nums, layers_dim=[16, 14, 8, 4]):
        super(OptimeA, self).__init__()
        self.optimA = MLP([A_nums+1] + layers_dim + [1])

    def forward(self, x, poisition):

        inputs = [x.transpose(1, 2), poisition.transpose(1, 2)]
        return self.optimA(torch.cat(inputs, dim=1))  # 输入为第一维方向拼接


class PredNN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.encoder = MLP([2] + layers + [1])
        # nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, x_pred, numbers):
        '''
        x_pred in R^24*1, number in R 24*1
        '''
        inputs = [x_pred, numbers]
        # 转换数据类型为float32
        return self.encoder(torch.cat(inputs, dim=1).to(torch.float32))

# class PredNN(nn.Module):
#     def __init__(self, in_features, layers_dim):
#         super(Net, self).__init__()
#         self.predict = MLP([in_features] + layers_dim + [1])

#     def forward(self, x):
#         return self.predict(x)


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
