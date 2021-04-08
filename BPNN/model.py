import torch
import torch.nn as nn
import torch.nn.functional as F


class MyBpNet(nn.Module):
    pass

    def __init__(self, in_features, out_features, layers):
        super(MyBpNet, self).__init__()
        self.num_inputs = in_features
        self.num_outputs = out_features

        self.layers = []
        self.in_layer = nn.Linear(self.num_inputs, layers[0])
        for i, item in enumerate(layers-2):
            self.layers.append(nn.Linear(item, layers[i+1]))
        self.out_layer = nn.Linear(self.layers[-1], out_features)

    def forward(self, A, x, norm=True):
        if norm is True:
            A = F.normalize(A, p=1, dim=-2)

        x = self.in_layer(x)
        x = F.relu(x)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = F.relu(x)
        x = self.out_layer(x)

        return x
        

            

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