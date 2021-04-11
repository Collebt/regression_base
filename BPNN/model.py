import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, in_features, out_features, layers_dim):
        super(Net, self).__init__()
        self.num_inputs = in_features
        self.num_outputs = out_features

        layers = []
        layers.append(nn.Linear(self.num_inputs, layers_dim[0]))
        layers.append(nn.ReLU())    
        for i, item in enumerate(layers_dim[:-1]):
            layers.append(nn.Linear(item, layers_dim[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layers_dim[-1], out_features))

        self.MLP = nn.Sequential(*layers)

    def forward(self, x):
        return self.MLP(x)
        
class Encoder(nn.Module):
    def __init__(self, in_feature, out_feature, layers_dim):
        super().__init__()
            

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