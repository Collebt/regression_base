import os
from easydict import EasyDict as edict
import numpy as np

__C = edict()
# j基础设置
cfg = __C

#模块路径
__C.MODULE = ''
__C.MODEL_NAME = ''
__C.DATASET_NAME = ''
__C.DATASET_FULL_NAME = 'A24'


# Minibatch size
__C.BATCH_SIZE = 4
__C.DATALOADER_NUM = 2


# Training options
__C.TRAIN = edict()

__C.TRAIN.START_EPOCH = 0
__C.TRAIN.NUM_EPOCHS = 15

__C.TRAIN.LR = 0.01
__C.TRAIN.MOMENTUM = 0.9
__C.TRAIN.LR_DECAY = 0.1
__C.TRAIN.LR_STEP = [10, 20]

__C.TRAIN.EPOCH_ITERS = 20
__C.TRAIN.LOSS_FUNC = 'offset'


# Evaluation options
__C.EVAL = edict()
__C.EVAL.EPOCH = 15
__C.EVAL.SAMPLES = 30


__C.Q = 1
__C.A = [34,17,8,45,33 ,44, 25, 13, 15,  9, 33, 22 ,16 ,35 ,41, 42, 18, 44, 29, 39, 17 ,26 ,32 , 8]
__C.NUMBER = 24
__C.LENGTH = 100
__C.RANDOM_SEED = 123









def _merge_a_into_b(a, b):
    '''
    融合自定义设置
    '''
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        if type(b[k]) is not type(v):
            raise ValueError(('Type mismatch ({} vs. {}) '
                              'for config key: {}').format(type(b[k]),
                                                           type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    '''
    通过加载yaml自定义文件
    '''
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
    """通过list加载自定义文件"""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d.keys()
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d.keys()
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value
