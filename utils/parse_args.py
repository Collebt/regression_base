import argparse
from utils.config import cfg, cfg_from_file, cfg_from_list
from pathlib import Path


def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--cfg', dest='cfg_file', action='append',
                        help='an optional config file', default=['experiments/net1.yaml'], type=str)
    parser.add_argument('--batch', dest='batch_size',
                        help='batch size', default=None, type=int)
    parser.add_argument('--epoch', dest='epoch',
                        help='epoch number', default=None, type=int)
    parser.add_argument('--model', dest='model',
                        help='model name', default=None, type=str)
    parser.add_argument('--dataset', dest='dataset',
                        help='dataset name', default=None, type=str)
    args = parser.parse_args()

    # load cfg from file
    if args.cfg_file is not None:
        for f in args.cfg_file:
            cfg_from_file(f)

    # load cfg from arguments
    if args.batch_size is not None:
        cfg_from_list(['BATCH_SIZE', args.batch_size])
    if args.epoch is not None:
        cfg_from_list(['TRAIN.START_EPOCH', args.epoch, 'EVAL.EPOCH', args.epoch])
    if args.model is not None:
        cfg_from_list(['MODEL_NAME', args.model])
    if args.dataset is not None:
        cfg_from_list(['DATASET_NAME', args.dataset])
        
    return args
