# -*- coding: utf-8 -*-
"""
@Time ： 2020/12/28 20:25
@Auth ： PUCHAZHONG
@File ：image_source.py
@IDE ：PyCharm
@CONTACT: zhonghw@zhejianglab.com
"""

import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList
import random, pdb, math, copy
from tqdm import tqdm
from loss import CrossEntropyLabelSmooth
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
import torch.nn.functional as F
from attentionconsistenct import AttentionConsistency
from aug_new import ACVCGenerator

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


def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_train_da(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.AugMix(),
        transforms.ToTensor(),
        normalize
    ])

def image_train_da1(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        # transforms.RandomCrop(crop_size),
        ACVCGenerator(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    # target
    txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_src = []
        for i in range(len(txt_src)):
            rec = txt_src[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.src_classes:
                line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
                new_src.append(line)
        txt_src = new_src.copy()

        new_tar = []
        for i in range(len(txt_test)):
            rec = txt_test[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'
                    new_tar.append(line)
        txt_test = new_tar.copy()
    # source 上 90% 训练集，10%验证
    if args.trte == "val":
        dsize = len(txt_src)
        tr_size = int(0.9 * dsize)
        # print(dsize, tr_size, dsize - tr_size)
        tr_txt, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
    # source 上 100% 训练集，10%验证
    else:
        dsize = len(txt_src)
        tr_size = int(0.9 * dsize)
        _, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
        tr_txt = txt_src

    dsets["source_tr"] = ImageList(tr_txt, transform=image_train())
    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True,
                                           num_workers=args.worker, drop_last=False)
    dsets["source_da"] = ImageList(tr_txt, transform=image_train_da())
    dset_loaders["source_da"] = DataLoader(dsets["source_da"], batch_size=train_bs, shuffle=True,
                                           num_workers=args.worker, drop_last=False)
    dsets["source_da1"] = ImageList(tr_txt, transform=image_train_da1())
    dset_loaders["source_da1"] = DataLoader(dsets["source_da1"], batch_size=train_bs, shuffle=True,
                                           num_workers=args.worker, drop_last=False)
    dsets["source_te"] = ImageList(te_txt, transform=image_test())
    dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs, shuffle=True,
                                           num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 2, shuffle=True, num_workers=args.worker,
                                      drop_last=False)

    return dset_loaders


def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            fea = netF(inputs)
            outputs = netC(netB(fea))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(all_output)).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy * 100, mean_ent


def cal_acc_oda(loader, netF, netB, netC, netC_ex, args, class_len, flag=False):
    start_test = True
    per_class_num = np.zeros((args.class_num + 1))
    per_class_correct = np.zeros((args.class_num + 1)).astype(np.float32) + 1e-9
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            outputs_ex = netC_ex(feas)
            # maxdummy, _ = torch.max(outputs_ex.clone(), dim=1)
            # maxdummy = maxdummy.view(-1, 1)

            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_output_ex = outputs_ex.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_output_ex = torch.cat((all_output_ex, outputs_ex.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    if flag:
        maxdummy, _ = torch.max(all_output_ex.clone(), dim=1)
        all_output_ex = maxdummy.view(-1, 1)
        all_output = torch.cat((all_output, all_output_ex),dim=1)
        all_output = nn.Softmax(dim=1)(all_output)
        _, predict = torch.max(all_output, 1)
        all_label = all_label.numpy()
        predict = predict.numpy()
        for i in range(len(all_label)):
            per_class_num[int(all_label[i])] += 1
            if all_label[i] == predict[i]:
                per_class_correct[int(all_label[i])] += 1
        pre_class_acc = per_class_correct / per_class_num * 100
        # if args.name1:
        #     a = 50 - args.tar
        start = args.class_num - class_len
        H_score = np.mean(pre_class_acc[start:-1]) * pre_class_acc[-1] * 2 / (pre_class_acc[-1] + np.mean(pre_class_acc[start:-1]))
        print("per_class_acc", pre_class_acc[start:], np.mean(pre_class_acc[start:]), np.mean(pre_class_acc[start:-1]), pre_class_acc[-1], H_score)
        # for i in label:
        #     if label[i] > 30:
        #         label[i] = 31
        # for i in predict:
        #     if predict[i] > 30:
        #         predict[i] = 31
        # predict[np.where(predict == iidx)[0]] = args.class_num
        # rev_label_map = {}
        # predict = predict.numpy()
        # label_list = list(set(predict))
        # label_list.sort()
        #
        # # label_list2 = torch.arange(0, args.class_num+1)
        # for i in range(len(label_list)):
        #     rev_label_map[label_list[i]] = i
        # for j in range(len(predict)):
        #     predict[j] = rev_label_map[predict[j]]
        # print("all_label", all_label, predict)
        # predict = torch.from_numpy(predict)
        # matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        # print(matrix, matrix.shape)
        # matrix = matrix[np.unique(all_label).astype(int), :]
        #
        # acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        # unknown_acc = acc[-1:].item()
        # H_score = np.mean(acc[:-1]) * unknown_acc * 2 / (unknown_acc + np.mean(acc[:-1]))
        # print("acc", acc, len(acc), "H_score", H_score)
        return np.mean(pre_class_acc[start:-1]), np.mean(pre_class_acc[start:]), pre_class_acc[-1]
        # return np.mean(acc[:-1]), np.mean(acc), unknown_acc
    # else:
    #     _, predict = torch.max(all_output, 1)
    #     all_output = nn.Softmax(dim=1)(all_output)
    #     re = torch.squeeze(predict).float()
    #     for i in range(len(all_label)):
    #         re[i] = torch.tensor(label_map[int(re[i])])
    #
    #     accuracy = torch.sum(re == all_label).item() / float(all_label.size()[0])
    #     mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    #     return accuracy * 100, mean_ent
def train_source(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    print(args.class_num)
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()
    netC_ex = network.feat_classifier(type=args.layer, class_num=args.extra_class, bottleneck_dim=args.bottleneck).cuda()
    modelpath = args.output_dir_src + '/source_F3.pt'
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B3.pt'
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C3.pt'
    netC.load_state_dict(torch.load(modelpath))

    param_group = []
    learning_rate = args.lr
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate * 0.1}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC_ex.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init = 0
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    interval_iter = max_iter // 10
    iter_num = 0

    netF.train()
    netB.train()
    netC.train()
    netC_ex.train()

    while iter_num < max_iter:
        try:
            inputs_source, labels_source = iter_source.next()
            inputs_source_da, _ = iter_source_da.next()
            inputs_source_da1, _ = iter_source_da1.next()
        except:
            torch.manual_seed(iter_num)
            iter_source = iter(dset_loaders["source_tr"])
            inputs_source, labels_source = iter_source.next()
            torch.manual_seed(iter_num)
            iter_source_da = iter(dset_loaders["source_da"])
            inputs_source_da, _ = iter_source_da.next()
            torch.manual_seed(iter_num)
            iter_source_da1 = iter(dset_loaders["source_da1"])
            inputs_source_da1, _ = iter_source_da1.next()

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_source, inputs_source_da, labels_source = inputs_source.cuda(), inputs_source_da.cuda(), labels_source.cuda()
        input = torch.cat((inputs_source, inputs_source_da), 0)
        fea = netF(input)

        outputs_source_n = netC(netB(fea))
        outputs_source_n_ex = netC_ex(netB(fea))
        # cam_0 = F.conv2d(feature,netB.bottleneck.weight.view(netB.bottleneck.out_features, feature.size(1),1,
        #             1)) + netB.bottleneck.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        # cam0 = F.conv2d(cam_0, netC.fc.weight.view(netC.fc.out_features, cam_0.size(1), 1, 1)) + netC.fc.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        # outputs_source_da = netC(netB(fea1))
        # cam_1 = F.conv2d(feature1, netB.bottleneck.weight.view(netB.bottleneck.out_features, feature1.size(1), 1,
        #                                               1)) + netB.bottleneck.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        # cam1 = F.conv2d(cam_1, netC.fc.weight.view(netC.fc.out_features, cam_1.size(1), 1,
        #                                           1)) + netC.fc.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        outputs_source, outputs_source_da = outputs_source_n.split(int(outputs_source_n.size(0)/2), dim = 0)
        outputs_source_ex, _ = outputs_source_n_ex.split(int(outputs_source_n.size(0) / 2), dim=0)
        outputs = torch.cat((outputs_source, outputs_source_ex), dim=1)
        maxdummy, _ = torch.max(outputs_source_ex.clone(), dim=1)
        maxdummy = maxdummy.view(-1, 1)
        dummpyoutputs = torch.cat((outputs_source.clone(), maxdummy), dim=1)
        outputs_source_negative = outputs_source.clone()
        for i in range(len(dummpyoutputs)):
            nowlabel = labels_source[i]
            dummpyoutputs[i][nowlabel] = -1e9
            outputs_source_negative[i][nowlabel] = -1e9

        dummytargets = torch.ones_like(labels_source) * args.class_num
        _, negative_label = torch.max(outputs_source_negative, dim=1)
        outputs_source_negative = torch.cat((outputs_source_negative, maxdummy), dim=1)
        outputs_source_negative = F.softmax(outputs_source_negative, 1)
        outputs_source_negative = 1 - outputs_source_negative
        label_n = torch.zeros((outputs_source_negative.size(0), outputs_source_negative.size(1))).long().cuda()
        label_range = torch.arange(0, outputs_source_negative.size(0)).long()
        label_n[label_range, negative_label] = 1
        # cam, cam1= cam0.split(int(cam0.size(0)/2), dim = 0)
        # cam1 = [cam1]
        entropy = -(torch.sum(label_n * torch.log(outputs_source_negative))) / float(outputs_source_negative.size(0))
        classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs, labels_source)
        classifier_loss += nn.CrossEntropyLoss()(dummpyoutputs, dummytargets) * 2

        classifier_loss += entropy

        p_clean, p_aug1 = F.softmax(outputs_source, dim=1), F.softmax(outputs_source_da, dim=1)

        # 计算平均值
        p_mixture = torch.clamp((p_clean + p_aug1 ) / 2., 1e-7, 1).log()
        # print("1", classifier_loss)
        classifier_loss += 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                      F.kl_div(p_mixture, p_aug1, reduction='batchmean')) / 2.
                      # F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
        # loss = AttentionConsistency(T=1.0)
        # classifier_loss += loss(cam, cam1, labels_source) * 0.5
        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            netC_ex.eval()

            if args.dset == 'VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['source_te'], netF, netB, netC, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, iter_num, max_iter,
                                                                            acc_s_te) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(dset_loaders['source_te'], netF, netB, netC, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, iter_num, max_iter, acc_s_te)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')

            # if acc_s_te >= acc_init:
            # acc_init = acc_s_te
            best_netF = netF.state_dict()
            best_netB = netB.state_dict()
            best_netC = netC.state_dict()
            best_netC_ex = netC_ex.state_dict()

            netF.train()
            netB.train()
            netC.train()
            netC_ex.train()

    torch.save(best_netF, osp.join(args.output_dir_src, "source_F5.pt"))
    torch.save(best_netB, osp.join(args.output_dir_src, "source_B5.pt"))
    torch.save(best_netC, osp.join(args.output_dir_src, "source_C5.pt"))
    torch.save(best_netC_ex, osp.join(args.output_dir_src, "source_C_ex5.pt"))

    return netF, netB, netC, netC_ex


def test_target(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()
    netC_ex = network.feat_classifier(type=args.layer, class_num=args.extra_class, bottleneck_dim=args.bottleneck).cuda()
    args.modelpath = args.output_dir_src + '/source_F5.pt'
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_B5.pt'
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_C5.pt'
    netC.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_C_ex5.pt'
    netC_ex.load_state_dict(torch.load(args.modelpath))
    netF.eval()
    netB.eval()
    netC.eval()
    netC_ex.eval()

    if args.da == 'oda':
        if True:
            acc_os1, acc_os2, acc_unknown = cal_acc_oda(dset_loaders['test'], netF, netB, netC, netC_ex,
                                                        args, 5, True)
            log_str1 = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}% / {:.2f}% / {:.2f}%'.format(args.name1, iter_num,
                                                                                             max_iter, acc_os2,
                                                                                             acc_os1, acc_unknown)
            print(log_str1)
            args.out_file.write(log_str1 + '\n')
            acc_os1, acc_os2, acc_unknown = cal_acc_oda(dset_loaders['test1'], netF, netB, netC, netC_ex,
                                                        args, 5, True)
            log_str1 = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}% / {:.2f}% / {:.2f}%'.format(args.name2, iter_num,
                                                                                             max_iter, acc_os2,
                                                                                             acc_os1, acc_unknown)
            print(log_str1)
            args.out_file.write(log_str1 + '\n')
            acc_os1, acc_os2, acc_unknown = cal_acc_oda(dset_loaders['test2'], netF, netB, netC, netC_ex,
                                                        args, 5, True)
            log_str1 = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}% / {:.2f}% / {:.2f}%'.format(args.name3, iter_num,
                                                                                             max_iter, acc_os2,
                                                                                             acc_os1, acc_unknown)
            print(log_str1)
            args.out_file.write(log_str1 + '\n')
    else:
        if args.dset == 'VISDA-C':
            acc, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, netC_ex, True)
            log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(args.trte, args.name, acc) + '\n' + acc_list
        else:
            acc, _ = cal_acc(dset_loaders['test'], netF, netB, netC, netC_ex, False)
            log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(args.trte, args.name, acc)

    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='1', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=50, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=32, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home',
                        choices=['VISDA-C', 'office31', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="vgg16, resnet50, resnet101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--extra_class', type=int, default=1)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='ckpt/source/')
    parser.add_argument('--da', type=str, default='oda', choices=['uda', 'pda', 'oda'])

    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
    args = parser.parse_args()

    if args.dset == 'office-home':
        # names = ['Real_World', 'Art', 'Product', 'Clipart']
        names = ['Real_World_U', 'Art_U', 'Product_U', 'Clipart_U']
        args.class_num = 15
    if args.dset == 'office31':
        names = ['webcam', 'amazon', 'dslr']
        args.class_num = 10
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10
    if args.dset == "cifar-stl":
        names = ['cifar9','stl9']
        args.class_num = 9


    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    folder = './data/'
    # args.s 选择源域是第几个数据集，default = 0 为 amazon
    # args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
    # args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
    args.s_dset_path = folder + args.dset + '/' + names[args.s] + '.txt'
    #print(args.s_dset_path)
    args.test_dset_path = folder + args.dset + '/' + names[args.t] + '.txt'
    if args.dset == 'office-home':
        if args.da == 'pda':
            args.class_num = 50
            args.src_classes = [i for i in range(50)]
            args.tar_classes = [i for i in range(25)]
        if args.da == 'oda':
            args.class_num = 15
            args.src_classes = [i for i in range(15)]
            args.tar_classes = [i for i in range(65)]
    if args.dset == 'office31':
        if args.da == 'oda':
            args.class_num = 10
            args.src_classes = [i for i in range(10)]
            args.tar_classes = [i for i in range(31)]
    # /ckpt/source/uda/office/A
    args.output_dir_src = osp.join(args.output, args.da, args.dset, names[args.s][0].upper())
    # 大写数据集首字母 （office 中为 A,D,W）
    args.name_src = names[args.s][0].upper()
    if not osp.exists(args.output_dir_src):
        os.system('mkdir -p ' + args.output_dir_src)
    if not osp.exists(args.output_dir_src):
        os.mkdir(args.output_dir_src)

    args.out_file = open(osp.join(args.output_dir_src, 'log.txt'), 'w')
    args.out_file.write(print_args(args) + '\n')
    args.out_file.flush()
    train_source(args)

    args.out_file = open(osp.join(args.output_dir_src, 'log_test.txt'), 'w')
    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i
        args.name = names[args.s][0].upper() + names[args.t][0].upper()

        folder = './data/'
        # args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
        # args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '.txt'
        if args.dset == 'office-home':
            if args.da == 'pda':
                args.class_num = 50
                args.src_classes = [i for i in range(50)]
                args.tar_classes = [i for i in range(25)]
            if args.da == 'oda':
                args.class_num = 15
                args.src_classes = [i for i in range(15)]
                args.tar_classes = [i for i in range(65)]
        if args.dset == 'office31':
            if args.da == 'oda':
                args.class_num = 10
                args.src_classes = [i for i in range(10)]
                args.tar_classes = [i for i in range(31)]
        args.name1 = names[0][0].upper() + names[1][0].upper()
        args.name2 = names[0][0].upper() + names[2][0].upper()
        args.name3 = names[0][0].upper() + names[3][0].upper()
        test_target(args)
