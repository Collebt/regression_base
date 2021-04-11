import torch
from torch.nn import DataParallel

'''
用于模型参数的保存和读取
'''

def save_model(model, path):
    if isinstance(model, DataParallel):
        model = model.module

    torch.save(model.state_dict(), path)


def load_model(model, path):
    if isinstance(model, DataParallel):
        model.module.load_state_dict(torch.load(path))
    else:
        model.load_state_dict(torch.load(path))