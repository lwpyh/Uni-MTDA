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

from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from sklearn.cluster import KMeans
from collections import Counter
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
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])


def image_test(resize_size=256, crop_size=224, alexnet=False):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()
    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                        drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 3, shuffle=False, num_workers=args.worker,
                                      drop_last=False)

    return dset_loaders


def cal_acc(loader, netF, netB, netC, label_list, label_map, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)), label_list)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1) / np.log(all_output.size(1))
    re = torch.squeeze(predict).float()
    for i in range(len(all_label)):
        re[i] = torch.tensor(label_map[int(re[i])])

    accuracy = torch.sum(re == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy * 100, mean_ent


def train_target(args):
    dset_loaders = data_load(args)
    ## set base network
    netF = network.ResBase(res_name=args.net).cuda()
    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier_dy(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()
    netC_ex =  network.feat_classifier(type=args.layer, class_num=1, bottleneck_dim=args.bottleneck).cuda()

    modelpath = args.output_dir_src + '/source_Fx.pt'
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_Bx.pt'
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_Cx.pt'
    netC.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C_exx.pt'
    netC_ex.load_state_dict(torch.load(modelpath))
    
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
            
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0
    label_map = {}
    rev_label_map = {}
    label_list = []
    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = next(iter_test)
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = next(iter_test)

        if iter_num % interval_iter == 0 and args.cls_par > 0:
            netF.eval()
            netB.eval()
            netC.eval()
            label_map = {}
            rev_label_map = {}
            mem_label = obtain_label1(dset_loaders['test'], netF, netB, netC, netC_ex, iter_num, args)
            label_list = list(set(mem_label))
            label_list.sort()
            for i in range(len(label_list)):
                label_map[i] = label_list[i]
                rev_label_map[label_list[i]] = i
            for j in range(len(mem_label)):
                mem_label[j] = rev_label_map[mem_label[j]]
            print(len(label_list), label_list)

            mem_label = torch.from_numpy(mem_label).cuda()
            netF.train()
            netB.train()
            netC.train()

        inputs_test = inputs_test.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        # bs*2048 , bs*1000
        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test, label_list)

        if args.cls_par > 0:
            pred = mem_label[tar_idx]
            classifier_loss = nn.CrossEntropyLoss()(outputs_test, pred)
            classifier_loss *= args.cls_par
        else:
            classifier_loss = torch.tensor(0.0).cuda()

        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                entropy_loss -= gentropy_loss
            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            if True:
                acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC,  label_list,label_map, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te)
                
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')
            netF.train()
            netB.train()
            netC.train()

    if args.issave:
        torch.save(netF.state_dict(), osp.join(args.output_dir, "target" + args.savename + ".pt"))

    return netF


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def get_thresh1(a):
    kmeans = KMeans(3, random_state=0).fit(a.reshape(-1,1))
    labels = kmeans.predict(a.reshape(-1, 1))
    idx = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]
    idx2 = np.where(labels == 2)[0]
    iidx = [idx, idx1, idx2][np.argmin([a[idx].mean(),a[idx1].mean(), a[idx2].mean()])]
    print(f"thresh: {max(a[iidx])}")
    return max(a[iidx])

def obtain_label1(loader, netF, netB, netC,netC_ex, iter_num, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas, list(range(args.class_num)))
            outputs_ex = netC_ex(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                all_output_ex = outputs_ex.float().cpu()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                all_output_ex = torch.cat((all_output_ex, outputs_ex.float().cpu()), 0)


    if iter_num == 0:
        dummy = torch.concat([all_output, all_output_ex], dim = 1)
        dummy = nn.Softmax(dim=1)(dummy)    
        _, predict_dummy = torch.max(dummy, 1)
        c = Counter(predict_dummy.numpy())
        cnt = sorted(c.items(), key=lambda k_v: k_v[1], reverse=True)
        if len(c) == 66:
            print(f"pda judge: class with least sample {cnt[-1]}")
        else:
            print(f"pda judge: class with least sample 0s")
        print(f"iter 0 : total_test: {predict_dummy.size()[0]}; num of openset_class {len(np.where(predict_dummy.cpu().numpy() == 65)[0])} ")
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    threshold = 0.0
    if iter_num == 0:
        threshold = 0.0
    elif min(cls_count)/max(cls_count) < 0.00001 :
        threshold = get_thresh1(cls_count)

    print(f"cls_count {cls_count}")
    labelset = np.where(cls_count >= threshold)
    labelset = labelset[0]
    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]
    guess_label = args.class_num * np.ones(len(all_label), )
    guess_label = pred_label
    acc = np.sum(guess_label == all_label.float().numpy()) / len(guess_label)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')
    return guess_label.astype('int')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='3', help="device id to run")
    parser.add_argument('--s', type=int, default=2, help="source")
    parser.add_argument('--t', type=int, default=0, help="target")
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
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--output', type=str, default='ckpt/target1106')
    parser.add_argument('--output_src', type=str, default='./Uni-MTDA/ckpt/source-ori')
    parser.add_argument('--da', type=str, default='pda', choices=['uda', 'pda', 'oda'])
    parser.add_argument('--issave', type=bool, default=False)
    args = parser.parse_args()

    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    if args.dset == 'office-home':
        names = ['Product', 'RealWorld', 'Art', 'Clipart']
        # names = ['Product', 'Art', 'Clipart', 'Real_World']
        # names = ['Clipart', 'Art', 'Product', 'Real_World']
        # names = ['Real_World', 'Product', 'Art', 'Clipart']
        args.class_num = 65
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
    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i

        folder = './data/office-home/'
        args.s_dset_path = folder + 'image_list/' + names[args.s] + '.txt'
        args.t_dset_path = folder + 'image_list/' + names[args.t] + '.txt'
        args.test_dset_path = folder + 'image_list/' + names[args.t] + '.txt'
        # args.s_dset_path = folder + args.dset + '/' + names[args.s] + '.txt'
        # args.t_dset_path = folder + args.dset + '/' + names[args.t] + '.txt'
        # args.test_dset_path = folder + args.dset + '/' + names[args.t] + '.txt'

        if args.dset == 'office-home':
            if args.da == 'pda':
                args.class_num = 65
                args.src_classes = [i for i in range(65)]
                args.tar_classes = [i for i in range(25)]

        args.output_dir_src = osp.join(args.output_src,"oda", args.dset, names[args.s][0].upper())
        args.output_dir = osp.join(args.output, args.da, args.dset, names[args.s][0].upper() + names[args.t][0].upper())
        args.name = names[args.s][0].upper() + names[args.t][0].upper()

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