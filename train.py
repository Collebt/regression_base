
from numpy.lib.function_base import average
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pathlib import Path
import time
from datetime import datetime

import matplotlib.pyplot as plt



from utils.config import cfg
from utils.parse_args import parse_args
from utils.model_sl import load_model, save_model

from data.data import *
from BPNN.loss import RegressionLoss





def train_model(model,
                optimizer,
                dataloader,
                num_epochs=25,
                resume=False,
                start_epoch=0):    
    print('Start training...')

    since = time.time()#记录时间开始节点
    dataset_size = len(dataloader['train'].dataset)

    #记录训练内存的设备
    device = next(model.parameters()).device 
    print('model on device: {}'.format(device))

    checkpoint_path = Path(cfg.OUTPUT_PATH) / 'params' #模型参数储存的位置
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True)

    if resume:#如果是继续训练模型，在现有参数中再进行优化
        assert start_epoch != 0
        model_path = str(checkpoint_path / 'params_{:04}.pt'.format(start_epoch))
        print('Loading model parameters from {}'.format(model_path))
        load_model(model, model_path)

        optim_path = str(checkpoint_path / 'optim_{:04}.pt'.format(start_epoch))
        print('Loading optimizer state from {}'.format(optim_path))
        optimizer.load_state_dict(torch.load(optim_path))
    
    record_loss = []
    record_acc = []

    #迭代训练模型
    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('_' * 10)

        model.train() #设置模型为训练模式（启动梯度传播）
        print('lr = ' + ', '.join(['{:.2e}'.format(x['lr']) for x in optimizer.param_groups]))
        
        epoch_loss = 0.0
        running_loss = 0.0
        running_since = time.time()
        iter_num = 0
        

        #读取样本数据
        for data in dataloader['train']:
            input_A = data['input_A']
            D = data['D']
            Q = data['Q']

            iter_num = iter_num + 1
            optimizer.zero_grad() #清空梯度

            with torch.set_grad_enabled(True):
                D_pred = model(input_A) #输入数据A，输出预测D

                loss = (D_pred - D) ** 2

                loss.backward() #反传参数
                optimizer.step() #优化器对参数进行梯度下降优化

                # statistics
                running_loss += loss.item() 
                epoch_loss += loss.item() 
                record_loss.append(loss.item())


                if iter_num % cfg.STATISTIC_STEP == 0:
                    running_speed = cfg.STATISTIC_STEP / (time.time() - running_since)
                    print('Epoch {:<4} Iteration {:<4} {:>4.2f}sample/s Loss={:<8.4f}'
                            .format(epoch, iter_num, running_speed, running_loss / cfg.STATISTIC_STEP ))
                    
                    running_loss = 0.0
                    running_since = time.time()
        epoch_loss = epoch_loss / dataset_size

        save_model(model, str(checkpoint_path / 'params_{:04}.pt'.format(epoch + 1)))
        torch.save(optimizer.state_dict(), str(checkpoint_path / 'optim_{:04}.pt'.format(epoch + 1)))

        print('Epoch {:<4} Loss: {:.4f}'.format(epoch, epoch_loss))
        print()
        #在每次迭代中验证效果
        accs, average_acc = eval_model(model, dataloader['test'])
        record_acc.append(average_acc.item())
        
    #plot

    fig, axs = plt.subplots(1, 2)
    axs[0].plot(np.array(record_acc))
    axs[0].set_title('average acc')

    axs[1].plot(np.array(record_loss))
    axs[1].set_title('loss')

    plt.savefig('train.png')


    return model


def eval_model(model, dataloader, eval_epoch=None, was_training=True):
    print('Start evaluation...')

    if eval_epoch is not None:
        model_path = str(Path(cfg.OUTPUT_PATH) / 'params' / 'params_{:04}.pt'.format(eval_epoch))
        print('Loading model parameters from {}'.format(model_path))
        load_model(model, model_path)

    # was_training = model.training
    model.eval() #切换成验证模式（不返传梯度）
    accs = []
    for data in dataloader:
        input_A = data['input_A']
        D = data['D']
        Q = data['Q']

        with torch.set_grad_enabled(False):
            D_pred = model(input_A) #输入数据A，输出预测D
            acc = 1 - torch.abs(D_pred / D- 1) #计算精度
            if was_training is False: #只有在最终验证的时候才输出每一个样本的精度
                print('predict / real : {:<8.4f} / {:<8.4f}, accary : {:<8.4f}'.format(D_pred.item(), D.item(), acc.item()))
            accs.append(acc)

            # statistics
    average_acc = torch.sum(torch.tensor(accs)) / cfg.EVAL.SAMPLES
    print('average accary:{:<8.4f}'.format(average_acc.item()))
            
    return accs, average_acc



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
    model = Net(in_features=cfg.NUMBER, out_features=1, layers_dim=cfg.LAYERS)

    #损失函数度量
    criterion = RegressionLoss()
    #优化器
    optimizer = optim.SGD(model.parameters(), lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM, nesterov=True)

    # print_easydict(cfg)
    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    # tfboardwriter = SummaryWriter(logdir=str(Path(cfg.OUTPUT_PATH) / 'tensorboard' / 'training_{}'.format(now_time)))

    model = train_model(model, optimizer, dataloader, 
                                 num_epochs=cfg.TRAIN.NUM_EPOCHS,
                                 resume=cfg.TRAIN.START_EPOCH != 0,
                                 start_epoch=cfg.TRAIN.START_EPOCH)
                            
    accs, average_acc = eval_model(model, dataloader['test'], was_training=False)

    