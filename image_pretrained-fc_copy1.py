import argparse
import os, sys
import os.path as osp
import torchvision
from torchvision import transforms
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from datasets import get_dataset


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
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()
    txt_tar1 = open(args.t_dset_path1).readlines()
    txt_test1 = open(args.test_dset_path1).readlines()
    txt_tar2 = open(args.t_dset_path2).readlines()
    txt_test2 = open(args.test_dset_path2).readlines()

    ##### openset sample 2 n+1 classes #######
    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar1)):
            rec = txt_tar1[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes1:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'
                    new_tar.append(line)
        txt_tar1 = new_tar.copy()
        txt_test1 = txt_tar1.copy()

        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_tar_n = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
                    new_tar_n.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'
                    new_tar_n.append(line)
        txt_tar = new_tar_n.copy()
        txt_test = txt_tar.copy()



    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                        drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 3, shuffle=False, num_workers=args.worker,
                                      drop_last=False)

    dsets["target1"] = ImageList_idx(txt_tar1, transform=image_train())
    dset_loaders["target1"] = DataLoader(dsets["target1"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                         drop_last=False)
    dsets["test1"] = ImageList_idx(txt_test1, transform=image_test())
    dset_loaders["test1"] = DataLoader(dsets["test1"], batch_size=train_bs * 3, shuffle=False, num_workers=args.worker,
                                       drop_last=False)

    dsets["target2"] = ImageList_idx(txt_tar2, transform=image_train())
    dset_loaders["target2"] = DataLoader(dsets["target2"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                         drop_last=False)
    dsets["test2"] = ImageList_idx(txt_test2, transform=image_test())
    dset_loaders["test2"] = DataLoader(dsets["test2"], batch_size=train_bs * 3, shuffle=False, num_workers=args.worker,
                                       drop_last=False)

    return dset_loaders


def cal_acc(loader, netF, netB, netC, netC_ex, label_list, label_map, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas, label_list)
            outputs_ex = netC_ex(feas)
            # outputs_ex, _ = torch.max(outputs_ex, 1)
            # outputs_ex = outputs_ex.view(-1, 1)
            # outputs = torch.cat((outputs, maxdummylogit), dim=1)
            if start_test:
                all_output = outputs.float().cpu()
                outputs_ex = outputs_ex.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_output_ex = torch.cat((all_output_ex, outputs_ex.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    if flag:
        all_output = torch.cat((all_output, all_output_ex),dim=1)
        all_output = nn.Softmax(dim=1)(all_output)
        # ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1) / np.log(args.class_num)
        #
        # from sklearn.cluster import KMeans
        # kmeans = KMeans(2, random_state=0).fit(ent.reshape(-1, 1))
        # labels = kmeans.predict(ent.reshape(-1, 1))

        # idx = np.where(labels == 1)[0]
        # iidx = 0
        # if ent[idx].mean() > ent.mean():
        #     iidx = 1
        _, predict = torch.max(all_output, 1)
        # predict[np.where(predict == iidx)[0]] = args.class_num

        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        matrix = matrix[np.unique(all_label).astype(int), :]

        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        unknown_acc = acc[-1:].item()
        return np.mean(acc[:-1]), np.mean(acc), unknown_acc
    else:
        _, predict = torch.max(all_output, 1)
        all_output = nn.Softmax(dim=1)(all_output)
        re = torch.squeeze(predict).float()
        for i in range(len(all_label)):
            re[i] = torch.tensor(label_map[int(re[i])])

        accuracy = torch.sum(re == all_label).item() / float(all_label.size()[0])
        mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
        return accuracy * 100, mean_ent


# def dynamic_fc_init(netC, param_group):
#     for k, v in netC.named_parameters():
#         param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
#     optimizer = optim.SGD(param_group)
#     optimizer = op_copy(optimizer)
#     return optimizer

def train_target(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier_dy(type=args.layer, class_num=50, bottleneck_dim=args.bottleneck).cuda()
    netC1 = network.feat_classifier_dy(type=args.layer, class_num=50, bottleneck_dim=args.bottleneck).cuda()
    netC2 = network.feat_classifier_dy(type=args.layer, class_num=50, bottleneck_dim=args.bottleneck).cuda()
    netC_ex = network.feat_classifier(type=args.layer, class_num=args.extra_class, bottleneck_dim=args.bottleneck).cuda()
    netC_ex1 = network.feat_classifier(type=args.layer, class_num=args.extra_class, bottleneck_dim=args.bottleneck).cuda()
    netC_ex2 = network.feat_classifier(type=args.layer, class_num=args.extra_class, bottleneck_dim=args.bottleneck).cuda()

    modelpath = args.output_dir_src + '/source_F3.pt'
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B3.pt'
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C3.pt'
    netC.load_state_dict(torch.load(modelpath))
    netC1.load_state_dict(torch.load(modelpath))
    netC2.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C_ex3.pt'
    netC_ex.load_state_dict(torch.load(modelpath))
    netC_ex1.load_state_dict(torch.load(modelpath))
    netC_ex2.load_state_dict(torch.load(modelpath))
    # netC.eval()
    for k, v in netC.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False
    # netCplus1 = network.Dy_Linear(50,50).cuda()
    for p, q in netC.named_parameters():
        param_group += [{'params': q, 'lr': args.lr * args.lr_decay1}]
    for p, q in netC1.named_parameters():
        param_group += [{'params': q, 'lr': args.lr * args.lr_decay1}]
    for p, q in netC2.named_parameters():
        param_group += [{'params': q, 'lr': args.lr * args.lr_decay1}]

    for p, q in netC_ex.named_parameters():
        param_group += [{'params': q, 'lr': args.lr * args.lr_decay1}]
    for p, q in netC_ex1.named_parameters():
        param_group += [{'params': q, 'lr': args.lr * args.lr_decay1}]
    for p, q in netC_ex2.named_parameters():
        param_group += [{'params': q, 'lr': args.lr * args.lr_decay1}]

    # netCplus2 = network.Dy_Linear(50,50).cuda()
    # for p, q in netCplus2.named_parameters():
    #     param_group += [{'params': q, 'lr': args.lr * args.lr_decay1}]
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    max_iter = args.max_epoch * max(len(dset_loaders["target"]), len(dset_loaders["target1"]), len(dset_loaders["target2"]))
    interval_iter = max_iter // args.interval
    iter_num = 0
    label_map = {}
    # rev_label_map = {}
    label_list = []
    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = iter_test.next()
            inputs_test1, _, tar_idx1 = iter_test1.next()
            inputs_test2, _, tar_idx2 = iter_test2.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()
            iter_test1 = iter(dset_loaders["target1"])
            inputs_test1, _, tar_idx1 = iter_test1.next()
            iter_test2 = iter(dset_loaders["target2"])
            inputs_test2, _, tar_idx2 = iter_test2.next()

        if inputs_test.size(0) == 1 or inputs_test1.size(0) == 1 or inputs_test2.size(0) == 1:
            continue

        if iter_num % interval_iter == 0 and args.cls_par > 0:
            netF.eval()
            netB.eval()
            netC.eval()
            netC1.eval()
            netC2.eval()
            netC_ex.eval()
            netC_ex1.eval()
            netC_ex2.eval()
            label_map = {}
            rev_label_map = {}
            mem_label = obtain_label(dset_loaders['test'], netF, netB, netC, netC_ex, args)
            label_list = list(set(mem_label))
            label_list.sort()
            for i in range(len(label_list)):
                label_map[i] = label_list[i]
                rev_label_map[label_list[i]] = i
            for j in range(len(mem_label)):
                mem_label[j] = rev_label_map[mem_label[j]]
            print(len(label_list))

            label_map1 = {}
            rev_label_map1 = {}
            mem_label1 = obtain_label(dset_loaders['test1'], netF, netB, netC1, netC_ex1, args)
            label_list1 = list(set(mem_label1))
            label_list1.sort()
            for i in range(len(label_list1)):
                label_map1[i] = label_list1[i]
                rev_label_map1[label_list1[i]] = i
            for j in range(len(mem_label1)):
                mem_label1[j] = rev_label_map1[mem_label1[j]]
            print(len(label_list1))

            label_map2 = {}
            rev_label_map2 = {}
            mem_label2 = obtain_label(dset_loaders['test2'], netF, netB, netC2, netC_ex2, args)
            label_list2 = list(set(mem_label2))
            label_list2.sort()
            for i in range(len(label_list2)):
                label_map2[i] = label_list2[i]
                rev_label_map2[label_list2[i]] = i
            for j in range(len(mem_label2)):
                mem_label2[j] = rev_label_map2[mem_label2[j]]
            print(len(label_list2))

            mem_label = torch.from_numpy(mem_label).cuda()
            mem_label1 = torch.from_numpy(mem_label1).cuda()
            mem_label2 = torch.from_numpy(mem_label2).cuda()

            acc_os1, acc_os2, acc_unknown = cal_acc(dset_loaders['test2'], netF, netB, netC2, netC_ex2, label_list2,
                                                    label_map2, True)
            log_str1 = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}% / {:.2f}% / {:.2f}%'.format(args.name, iter_num,
                                                                                             max_iter, acc_os2,
                                                                                             acc_os1, acc_unknown)
            print(log_str1)

            netF.train()
            netC1.train()
            netC2.train()
            netB.train()
            netC.train()
            netC_ex.train()
            netC_ex1.train()
            netC_ex2.train()

        inputs_test = inputs_test.cuda()
        inputs_test1 = inputs_test1.cuda()
        inputs_test2 = inputs_test2.cuda()

        input_test = torch.cat([inputs_test, inputs_test1, inputs_test2], dim=0)
        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        # bs*2048 , bs*1000
        features_test = netB(netF(input_test))
        # outputs_test = netC(features_test)
        outputs_test1 = netC((features_test[:len(tar_idx), ...]), label_list)
        outputs_test2 = netC1(features_test[len(tar_idx):len(tar_idx)+len(tar_idx1), ...], label_list1)
        outputs_test3 = netC2(features_test[len(tar_idx)+len(tar_idx1):, ...], label_list2)

        if args.cls_par > 0:
            pred = mem_label[tar_idx]
            pred1 = mem_label1[tar_idx1]
            pred2 = mem_label2[tar_idx2]
            classifier_loss = nn.CrossEntropyLoss()(outputs_test1, pred) + nn.CrossEntropyLoss()(outputs_test2, pred1) + nn.CrossEntropyLoss()(outputs_test3, pred2)
            classifier_loss *= args.cls_par

        else:
            classifier_loss = torch.tensor(0.0).cuda()
        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_test1)
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            softmax_out1 = nn.Softmax(dim=1)(outputs_test2)
            entropy_loss1 = torch.mean(loss.Entropy(softmax_out1))
            softmax_out2 = nn.Softmax(dim=1)(outputs_test3)
            entropy_loss2 = torch.mean(loss.Entropy(softmax_out2))
            entropy_loss = entropy_loss + entropy_loss1 + entropy_loss2
            if args.gent:
                msoftmax = softmax_out.mean(0)
                msoftmax1 = softmax_out1.mean(0)
                msoftmax2 = softmax_out2.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                gentropy_loss1 = torch.sum(-msoftmax1 * torch.log(msoftmax1 + args.epsilon))
                gentropy_loss2 = torch.sum(-msoftmax2 * torch.log(msoftmax2 + args.epsilon))
                gentropy_loss = gentropy_loss + gentropy_loss1 + gentropy_loss2
                # gentropy_loss = torch.sum((msoftmax - idx_1k) ** 2) / torch.sum(idx_1k)
                # print(gentropy_loss.cpu().item())
                entropy_loss -= gentropy_loss

            # print(entropy_loss.cpu().item(), classifier_loss.cpu().item())
            classifier_loss += entropy_loss * args.ent_par
        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            netC1.eval()
            netC2.eval()
            netC_ex.eval()
            netC_ex1.eval()
            netC_ex2.eval()
            if args.name == args.name3:
                acc_os1, acc_os2, acc_unknown = cal_acc(dset_loaders['test2'], netF, netB, netC2, netC_ex2,label_list2, label_map2, True)
                log_str1 = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}% / {:.2f}% / {:.2f}%'.format(args.name, iter_num,
                                                                                                max_iter, acc_os2,
                                                                                                acc_os1, acc_unknown)
                args.out_file.write(log_str1 + '\n')
            else:
                acc1, _ = cal_acc(dset_loaders['test'], netF, netB, netC, netC_ex, label_list, label_map)
                # acc, ment = cal_acc(dset_loaders['test'], netF,netC, netCplus1, label_list,label_map)
                log_str1 = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name1, iter_num, max_iter, acc1)
                args.out_file.write(log_str1 + '\n')
                acc2, _ = cal_acc(dset_loaders['test1'], netF, netB, netC1, netC1_ex, label_list1, label_map1)
                log_str2 = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name2, iter_num, max_iter, acc2)
                args.out_file.write(log_str2 + '\n')
            args.out_file.flush()
            # print(log_str + '\n')
            netF.train()
            netB.train()
            netC.train()
            netC1.train()
            netC2.train()
            netC_ex.train()
            netC_ex1.train()
            netC_ex2.train()

    if args.issave:
        torch.save(netF.state_dict(), osp.join(args.output_dir, "target" + args.savename + ".pt"))

    return netF


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def get_thresh(a):
    suitable_th = 0
    max_g = 0
    for i in range(int(min(a)), int(max(a)) + 1):
        sA = a > i
        sB = a <= i
        fA = np.sum(sA)
        fB = np.sum(sB)
        if 0 == fA:
            break
        if 0 == fB:
            continue

        w0 = float(fA) / 1000.
        u0 = float(np.sum(a * sA)) / fA
        w1 = float(fB) / 1000.
        u1 = float(np.sum(a * sB)) / fB

        g = w0 * w1 * (u0 - u1) * (u0 - u1)
        if g > max_g:
            max_g = g
            suitable_th = i
    return suitable_th


def obtain_label(loader, netF, netB, netC, netC_ex, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas, list(range(50)))
            outputs_ex = netC_ex(feas)
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

    all_output = torch.cat((all_output, all_output_ex), dim=1)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    known_idx = np.where(predict != args.class_num)[0]

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea[known_idx].float().cpu().numpy()
    all_output = all_output[known_idx, :]
    predict = predict[known_idx]
    all_label_idx = all_label[known_idx]

    print("K", all_output.size(1))
    K = all_output.size(1) - 1
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    threshold = 0.0
    if min(cls_count[:])/max(cls_count[:]) < 0.01:
        threshold = get_thresh(cls_count)
    print("cls_count", cls_count, threshold, min(cls_count[:]), max(cls_count[:]), min(cls_count[:])/max(cls_count[:-1]))
    labelset = np.where(cls_count >= threshold)
    labelset = labelset[0]
    print(labelset)

    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]
    # print(pred_label[:50])
    guess_label = args.class_num * np.ones(len(all_label), )
    guess_label[known_idx] = pred_label
    print("test_part", guess_label, all_label, len(guess_label[known_idx]), len(all_label[known_idx]), len(all_label_idx))
    if args.name == args.name3:
        acc = np.sum(guess_label[known_idx] == all_label.float().numpy()) / float(len(all_label_idx))
    else:
        acc = np.sum(guess_label[known_idx] == all_label[known_idx].float().numpy()) / float(len(all_label))
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')

    return guess_label.astype('int')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')

    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=15, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home',
                        choices=['VISDA-C', 'office', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")

    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=10)
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)
    parser.add_argument('--extra_class', type=int, default=1)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--output', type=str, default='ckpt/target')
    parser.add_argument('--output_src', type=str, default='ckpt/source')
    parser.add_argument('--da', type=str, default='pda', choices=['uda', 'pda'])
    parser.add_argument('--issave', type=bool, default=False)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    if args.dset == 'office-home':
        names = ['Clipart', 'Real_World', 'Product', 'Art']
        args.class_num = 50
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10
    if args.dset == "cifar-stl":
        names = ['cifar9', 'stl9']
        args.class_num = 9
    args.s = 0
    folder = './data/'
    args.s_dset_path = folder + args.dset + '/' + names[0] + '.txt'
    # print(args.s_dset_path)
    args.t_dset_path = folder + args.dset + '/' + names[1] + '.txt'
    args.test_dset_path = folder + args.dset + '/' + names[1] + '.txt'
    args.t_dset_path1 = folder + args.dset + '/' + names[2] + '.txt'
    args.test_dset_path1 = folder + args.dset + '/' + names[2] + '.txt'
    args.t_dset_path2 = folder + args.dset + '/' + names[3] + '.txt'
    args.test_dset_path2 = folder + args.dset + '/' + names[3] + '.txt'
    # 输出路径
    # ckpt/source/uda/office/A
    args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())
    print(args.output_dir_src)
    args.output_dir = osp.join(args.output, args.da, args.dset + "-addfc+gent")
    args.name = args.dset
    if args.dset == 'office-home':
        args.class_num = 50
        args.src_classes = [i for i in range(50)]
        args.tar_classes = [i for i in range(50)]
        args.tar_classes1 = [i for i in range(50)]
        args.tar_classes2 = [i for i in range(65)]
    args.name1 = names[0][0].upper() + names[1][0].upper()
    args.name2 = names[0][0].upper() + names[2][0].upper()
    args.name3 = names[0][0].upper() + names[3][0].upper()

    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.savename = 'par_' + str(args.cls_par)
    if args.da == 'pda':
        args.savename = 'par_' + str(args.cls_par) + '_thr' + str(args.threshold)
    args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
    args.out_file.write(print_args(args) + '\n')
    args.out_file.flush()
    print(args)
    train_target(args)