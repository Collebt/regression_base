
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.config import cfg
from utils.parse_args import parse_args

from data.data import *
from BPNN.loss import RegressionLoss


def train_model(model,
                criterion,
                optimizer,
                dataloader,
                tfboard_writer,
                num_epochs=25,
                resume=False,
                start_epoch=0):    
    print('Start training...')




if __name__ == '__main__':
    args = parse_args('bp网络训练代码')

    import importlib
    mod = importlib.import_module(cfg.MODULE)
    Net = mod.Net

    #构建数据集
    dataset_len = {'train': cfg.TRAIN.EPOCH_ITERS * cfg.BATCH_SIZE, 'test': cfg.EVAL.SAMPLES}
    A_dataset = {
        x: MyDataset(   cfg.DATASET_FULL_NAME,
                        sets=x,
                        Q=cfg.Q,
                        number=cfg.NUMBER,
                        length=dataset_len[x],
                        A = cfg.A
                        )
                        for x in ('train', 'test')
    }
    dataloader = {x: get_dataloader(A_dataset[x]) for x in ('train', 'test')} #构建为可迭代的数据片

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #实体化网络
    model = Net(in_features=cfg.NUMBER, out_features=1, layers_dim=[32, 64, 32, 24])

    #损失函数度量
    criterion = RegressionLoss()
    #优化器
    optimizer = optim.SGD(model.parameters(), lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM, nesterov=True)

    # print_easydict(cfg)

    model = train_model(model, criterion, optimizer, dataloader, tfboardwriter,
                                 num_epochs=cfg.TRAIN.NUM_EPOCHS,
                                 resume=cfg.TRAIN.START_EPOCH != 0,
                                 start_epoch=cfg.TRAIN.START_EPOCH)