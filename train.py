
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config import cfg
from data import *
from loss import RegressionLoss
from parse_args import parse_args


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
    from model import MyBpNet

    args = parse_args('bp网络训练代码')

    import importlib
    mod = importlib.import_module(cfg.MODULE)
    Net = mod.Net

    #构建数据集
    dataset_len = {'train': cfg.TRAIN.EPOCH_ITERS * cfg.BATCH_SIZE, 'test': cfg.EVAL.SAMPLES}
    A_dataset = {
        x: MyDataset(   Q=cfg.Q,
                        number=cfg.NUMBER,
                        length=dataset_len[x]
                        )
                        for x in ('train', 'test')
    }
    dataloader = {x: get_dataloader(A_dataset[x]) for x in ('train', 'test')}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #实体化网络
    model = Net()

    #损失函数度量
    criterion = RegressionLoss()
    #优化器
    optimizer = optim.SGD(model.parameters(), lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM, nesterov=True)

    # print_easydict(cfg)

    model = train_model(model, criterion, optimizer, dataloader, tfboardwriter,
                                 num_epochs=cfg.TRAIN.NUM_EPOCHS,
                                 resume=cfg.TRAIN.START_EPOCH != 0,
                                 start_epoch=cfg.TRAIN.START_EPOCH)