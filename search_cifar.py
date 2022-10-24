# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import time
from argparse import ArgumentParser

import torch
import torch.nn as nn
import sys
sys.path.append("../../../nni")
import datasets
from model import CNN,feat_bootleneck,feat_classifier
from nni.nas.pytorch.callbacks import ArchitectureCheckpoint, LRSchedulerCallback
from nni.algorithms.nas.pytorch.dartshot import DartsTrainer
from utils import accuracy
import numpy as np
import random
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
logger = logging.getLogger('nni')
import torch.optim as optim
from torchvision import transforms



def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer



# def image_train(resize_size=256, crop_size=224, alexnet=False):
#     if not alexnet:
#         normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                          std=[0.229, 0.224, 0.225])
#     else:
#         normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
#     return transforms.Compose([
#         transforms.Resize((resize_size, resize_size)),
#         transforms.RandomCrop(crop_size),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         normalize
#     ])
#
#
# def image_test(resize_size=256, crop_size=224, alexnet=False):
#     if not alexnet:
#         normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                          std=[0.229, 0.224, 0.225])
#     else:
#         normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
#     return transforms.Compose([
#         transforms.Resize((resize_size, resize_size)),
#         transforms.CenterCrop(crop_size),
#         transforms.ToTensor(),
#         normalize
#     ])

def data_load():
    ## prepare data
    train_bs = 64
    train_stl, val_stl = datasets.get_dataset("STL10")
    train_loader = DataLoader(train_stl, batch_size=train_bs, shuffle=True,
                                           num_workers=4, drop_last=False)
    valid_loader = DataLoader(train_stl, batch_size=train_bs * 3, shuffle=False, num_workers=4,
                                      drop_last=False)

    return train_loader,valid_loader

if __name__ == "__main__":
    parser = ArgumentParser("darts")
    parser.add_argument("--layers", default = 8, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--log-frequency", default=10, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--channels", default=16, type=int)
    parser.add_argument("--unrolled", default=False, action="store_true")
    parser.add_argument("--visualization", default=False, action="store_true")
    parser.add_argument('--output', type=str, default='ckpt/target')
    parser.add_argument('--output_src', type=str, default='ckpt/source')
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    args = parser.parse_args()
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    torch.manual_seed(666)
    torch.cuda.manual_seed(666)
    np.random.seed(666)
    random.seed(666)
    # dataset_train, dataset_valid = datasets.get_dataset("cifar10")
    train_loader, valid_loader = data_load()
    model_base = CNN(32, 3, args.channels, 9, args.layers)
    netB = feat_bootleneck(type="bn", feature_dim=2048,
                                   bottleneck_dim=256).cuda()
    netC = feat_classifier(type="wn", class_num=9, bottleneck_dim=256).cuda()
    modelpath = "ckpt/source/uda/cifar-stl/C" + '/source_B.pt'
    netB.load_state_dict(torch.load(modelpath))
    modelpath = "ckpt/source/uda/cifar-stl/C" + '/source_C.pt'
    netC.load_state_dict(torch.load(modelpath))
    netC.eval()
    for k, v in netC.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netB.named_parameters():
        # if args.lr_decay2 > 0:
        param_group += [{'params': v, 'lr': args.lr * 1.0}]
        # else:
        #     v.requires_grad = False
    for k,v in model_base.named_parameters():
        param_group += [{'params': v, 'lr': args.lr * 1.0}]
    # print(model.out_feature)
    criterion = nn.CrossEntropyLoss()

    # optim = torch.optim.SGD(param_group, 0.025, momentum=0.9, weight_decay=3.0E-4)

    optimizer = optim.SGD(param_group)
    # lr_scheduler =
    optimizer = op_copy(optimizer)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0.001)
    trainer = DartsTrainer(model_base, netB, netC,
                           loss=criterion,
                           metrics=lambda output, target: accuracy(output.cpu(), target.cpu(), topk=(1,)),
                           optimizer=optimizer,
                           num_epochs=args.epochs,
                           dataset_train=train_loader,
                           dataset_valid=valid_loader,
                           batch_size=args.batch_size,
                           log_frequency=args.log_frequency,
                           unrolled=args.unrolled,
                           callbacks=[LRSchedulerCallback(lr_scheduler), ArchitectureCheckpoint("./checkpoints")])
    if args.visualization:
        trainer.enable_visualization()
    trainer.train()
