import numpy as np
import torch
import torch.optim as optim

from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt

from utils.config import cfg
from utils.parse_args import parse_args
from utils.model_sl import load_model, save_model

from data.data import *


def train_model(model,
                optimizer,
                dataloader,
                num_epochs=25,
                resume=False,
                start_epoch=0):
    print('训练网络...')

    dataset_size = len(dataloader['train'].dataset)

    checkpoint_path = Path(cfg.OUTPUT_PATH) / 'params'  # 模型参数储存的位置
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True)

    if resume:  # 如果是继续训练模型，在现有参数中再进行优化
        assert start_epoch != 0
        model_path = str(checkpoint_path /
                         'params_{:04}.pt'.format(start_epoch))
        print('Loading model parameters from {}'.format(model_path))
        load_model(model, model_path)

        optim_path = str(checkpoint_path /
                         'optim_{:04}.pt'.format(start_epoch))
        print('Loading optimizer state from {}'.format(optim_path))
        optimizer.load_state_dict(torch.load(optim_path))

    record_loss = []
    record_acc = []

    model.to(torch.float32)  # 将模型参数转换为float32格式

    # 迭代训练模型
    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('_' * 10)

        model.train()  # 设置模型为训练模式（启动梯度传播）
        print('lr = ' + ', '.join(['{:.2e}'.format(x['lr'])
              for x in optimizer.param_groups]))

        epoch_loss = 0.0
        running_loss = 0.0
        iter_num = 0

        # 读取样本数据
        for data in dataloader['train']:
            input_A = data['A_input']
            A_pos = data['A_pos']
            A_nums = data['A_nums']
            A_better = data['A_better']
            D = data['D']
            Q = data['Q']

            iter_num = iter_num + 1
            optimizer.zero_grad()  # 清空梯度

            with torch.set_grad_enabled(True):
                Q_pred, A_pred = model(
                    input_A, A_pos, A_nums)  # 输入数据A，输出预测Q和预测的A

                loss = torch.sum((A_pred - A_better) ** 2) + \
                    (Q_pred - D) ** 2  # 损失函数

                loss.backward()  # 反传参数
                optimizer.step()  # 优化器对参数进行梯度下降优化

                # statistics
                running_loss += loss.item()
                epoch_loss += loss.item()
                record_loss.append(loss.item())

                if iter_num % cfg.STATISTIC_STEP == 0:
                    print('Epoch {:<4} Iteration {:<4} Loss={:<8.4f}'
                          .format(epoch, iter_num, running_loss / cfg.STATISTIC_STEP))

                    running_loss = 0.0
        epoch_loss = epoch_loss / dataset_size

        save_model(model, str(checkpoint_path /
                   'params_{:04}.pt'.format(epoch + 1)))
        torch.save(optimizer.state_dict(), str(
            checkpoint_path / 'optim_{:04}.pt'.format(epoch + 1)))

        print('Epoch {:<4} Loss: {:.4f}'.format(epoch, epoch_loss))
        print()
        # 在每次迭代中验证效果
        accs, average_acc = eval_model(model, dataloader['test'])
        record_acc.append(average_acc.item())

    # plot
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(np.array(record_acc))
    axs[0].set_title('average acc')

    axs[1].plot(np.array(record_loss))
    axs[1].set_title('loss')

    plt.savefig('train.png')

    return model


def eval_model(model, dataloader, eval_epoch=None, was_training=True):
    print('验证网络')

    if eval_epoch is not None:
        model_path = str(Path(cfg.OUTPUT_PATH) / 'params' /
                         'params_{:04}.pt'.format(eval_epoch))
        print('Loading model parameters from {}'.format(model_path))
        load_model(model, model_path)

    # was_training = model.training
    model.eval()  # 切换成验证模式（不返传梯度）
    accs = []
    A_accs = []
    for data in dataloader:
        input_A = data['A_input']
        A_pos = data['A_pos']

        A_nums = data['A_nums']
        A_better = data['A_better']
        D = data['D']
        Q = data['Q']

        with torch.set_grad_enabled(False):
            # 输入数据A，输出预测D和预测A
            Q_pred, A_pred = model(input_A, A_pos, A_nums)

            acc = 1 - torch.abs(Q_pred / D - 1)  # 计算精度
            A_acc = torch.mean(1 - torch.abs(A_pred / A_better - 1))
            if was_training is False:  # 只有在最终验证的时候才输出每一个样本的精度
                print('predict / real : {:<8.4f} / {:<8.4f}, accary : {:<8.4f}, acc_A : {:8.4f}'.format(
                    Q_pred.item(), D.item(), acc.item(), A_acc.item()))

            accs.append(acc)
            A_accs.append(A_acc)

            # statistics
    average_acc = torch.sum(torch.tensor(accs)) / cfg.EVAL.SAMPLES
    average_A_acc = torch.sum(torch.tensor(A_accs)) / cfg.EVAL.SAMPLES
    print('average accary:{:<8.4f}, acc_A:{:<8.4f}'.format(
        average_acc.item(), average_A_acc.item()))

    return accs, average_acc


if __name__ == '__main__':
    args = parse_args('bp网络训练代码')  # 初始化设置和自定义

    import importlib
    mod = importlib.import_module(cfg.MODULE)
    Net = mod.Net

    # 构建数据集，分为训练集和验证集
    dataset_len = {'train': cfg.TRAIN.EPOCH_ITERS *
                   cfg.BATCH_SIZE, 'test': cfg.EVAL.SAMPLES}
    A_dataset = {
        x: MyDataset(cfg.DATASET_FULL_NAME,
                     sets=x,
                     Q=cfg.Q,
                     number=cfg.NUMBER,
                     length=dataset_len[x],
                     A=cfg.A,
                     A_better=cfg.A_better,
                     dummy=cfg.DATA.DUMMY
                     )
        for x in ('train', 'test')
    }
    dataloader = {x: get_dataloader(A_dataset[x]) for x in (
        'train', 'test')}  # 构建为可迭代的数据加载器

    # 实体化网络
    model = Net(A_nums=cfg.NUMBER, out_features=1,
                optim_layers=cfg.OPTIMLAYERS, predict_layers=cfg.PREDLAYERS)

    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=cfg.TRAIN.LR,
                          momentum=cfg.TRAIN.MOMENTUM, nesterov=True)

    # 记录目前时间
    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    # 训练模型
    model = train_model(model, optimizer, dataloader,
                        num_epochs=cfg.TRAIN.NUM_EPOCHS,
                        resume=cfg.TRAIN.START_EPOCH != 0,
                        start_epoch=cfg.TRAIN.START_EPOCH)

    # 验证模型
    accs, average_acc = eval_model(
        model, dataloader['test'], was_training=False)

    # 测试A的精度
    input_A = torch.tensor(cfg.A).reshape(1, -1, 1)
    A_pos = torch.eye(cfg.NUMBER).reshape(1, cfg.NUMBER, -1)
    A_nums = torch.ones(1, cfg.NUMBER).reshape(1, -1, 1)
    Q_pred, A_pred = model(input_A, A_pos, A_nums)
    print('predict A:', A_pred.reshape(-1))
    print('real A:', cfg.A_better)
