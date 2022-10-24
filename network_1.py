# -*- coding: utf-8 -*-
"""
@Time ： 2020/12/29 9:56
@Auth ： PUCHAZHONG
@File ：network.py
@IDE ：PyCharm
@CONTACT: zhonghw@zhejianglab.com
"""
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import torch.nn.utils.weight_norm as weightNorm
from collections import OrderedDict
from torch.nn import functional as F
from torch.nn import init
from torch.nn.parameter import Parameter

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

vgg_dict = {"vgg11":models.vgg11, "vgg13":models.vgg13, "vgg16":models.vgg16, "vgg19":models.vgg19,
"vgg11bn":models.vgg11_bn, "vgg13bn":models.vgg13_bn, "vgg16bn":models.vgg16_bn, "vgg19bn":models.vgg19_bn}
class VGGBase(nn.Module):
  def __init__(self, vgg_name):
    super(VGGBase, self).__init__()
    model_vgg = vgg_dict[vgg_name](pretrained=True)
    self.features = model_vgg.features
    self.classifier = nn.Sequential()
    for i in range(6):
        self.classifier.add_module("classifier"+str(i), model_vgg.classifier[i])
    self.in_features = model_vgg.classifier[6].in_features

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x

res_dict = {"resnet18":models.resnet18, "resnet34":models.resnet34, "resnet50":models.resnet50,
"resnet101":models.resnet101, "resnet152":models.resnet152, "resnext50":models.resnext50_32x4d, "resnext101":models.resnext101_32x8d}

class ResBase(nn.Module):
    def __init__(self, res_name):
        super(ResBase, self).__init__()
        model_resnet = res_dict[res_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features
        self.fc = nn.Linear(2048, 50)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        feature = x.clone()
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # cam = F.conv2d(feature, self.fc.weight.view(self.fc.out_features,
        #             feature.size(1), 1, 1)) + self.fc.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        return x, feature

class feat_bootleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(feat_bootleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x

class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x


class feat_classifier_dy(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(feat_classifier_dy, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(Dy_Linear(bottleneck_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = Dy_Linear(bottleneck_dim, class_num)
            self.fc.apply(init_weights)

    def forward(self, x, label_map):
        x = self.fc(x, label_map)
        return x


class feat_classifier_two(nn.Module):
    def __init__(self, class_num, input_dim, bottleneck_dim=256):
        super(feat_classifier_two, self).__init__()
        self.type = type
        self.fc0 = nn.Linear(input_dim, bottleneck_dim)
        self.fc0.apply(init_weights)
        self.fc1 = nn.Linear(bottleneck_dim, class_num)
        self.fc1.apply(init_weights)

    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        return x

class Res50(nn.Module):
    def __init__(self):
        super(Res50, self).__init__()
        model_resnet = models.resnet50(pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features
        self.fc = model_resnet.fc


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        return x, y



class Dy_Linear(nn.Module):

    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(Dy_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features


        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input, label_map):
        label_map.sort()
        self.label_map = torch.LongTensor(label_map).cuda()
        if not self.label_map.size():
            return F.linear(input, self.weight, self.bias)
        else:
            self.label_map.sort()
            temp_w = self.weight.index_select(0,self.label_map)
            temp_b = self.bias.index_select(0,self.label_map)
            return F.linear(input, temp_w, temp_b)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


#
# class Dynamic_fc(nn.Module):
#     def __init__(self, label_map):
#         super(Dynamic_fc, self).__init__()
#         self.dynamic_fc = nn.Linear(1000, 1000)
#         self.dynamic_fc.apply(init_weights)
#         self.tune(label_map)
#     def forward(self, x):
#         return self.dynamic_fc(x)
#     def tune(self,label_map):
#         self.new_fc = nn.Linear(1000, len(label_map))
#         temp_w =
#         for key in self.new_fc.state_dict().keys():
#
#         self.new_fc.parameters().


class Dynamic_conv(nn.Module):
    def __init__(self,num_class):
        super(Dynamic_conv, self).__init__()
        self.dynamic_conv = nn.Conv2d(2048, 1000, [1,1])
    def forward(self, x):
        y = self.dynamic_conv(x)
        y = y.view(y.size(0), -1)
        return y
    # def tune(self, label_list):
        # for name,parameters in self.dynamic_conv.named_parameters():


if __name__ =="__main__":
    torch.manual_seed(666)
    a = Dy_Linear(4,4)
    label_map = [0,1,2]
    x = torch.rand((2,4))

    # print(c)
    opt = torch.optim.SGD(a.parameters(),lr = 0.1)
    for i in range(4):
        c = a(x, label_map)
        loss = torch.sum(torch.sum(c))
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(a.state_dict())

    # x = torch.tensor((10,1000))



    # d = nn.Linear(4,2)

    # parm = {}
    # state_dict = d.state_dict()
    # state_dict['weight'] = temp_w[:2,...]
    # d.load_state_dict(state_dict)
    # for name, parameters in d.named_parameters():
    #     print(name, ':', parameters.size())
    #     parm[name] = parameters.detach().numpy()

    #
    # x = torch.rand(5,4)
    #
    #
    #
    # for i in range(5):
    #     print('a_init')
    #     print(a.state_dict())
    #     temp_w = a.state_dict()['weight']
    #     temp_b = a.state_dict()['bias']
    #
    #     n = random.randint(1, 4)
    #     d = nn.Linear(4,n)
    #     state_dict = d.state_dict()
    #     state_dict['weight'] = temp_w[:n, ...]
    #     state_dict['bias'] = temp_b[:n,...]
    #     d.load_state_dict(state_dict)
    #     opti = torch.optim.SGD(d.parameters(), lr=0.1)
    #     opti.zero_grad()
    #     y = d.forward(x)
    #     loss = y[0][0]
    #     loss.backward()
    #     opti.step()
    #     new_a = a.state_dict()
    #     new_a['weight'][:n,...] = d.state_dict()['weight']
    #     new_a['bias'][:n,...] = d.state_dict()['bias']
    #     a.load_state_dict(new_a)
    #     print(str(i)+'\n  a:')
    #     print(a.state_dict())
    #     print(d.state_dict())
    #     # for name,parameters in d.named_parameters():
    #     #     print(i)
    #     #     print(name,':',parameters.size())
    #     #     parm[name] = parameters.detach().numpy()
    #     # print(parm)
    # print(a.state_dict())

    # print(list(a.items()))
    # print(d.parameters())
    # d["weight"].copy_(x)
    # for name,parameters in d.named_parameters():
    #
    #     print(name,':',parameters.size())
    #     parm[name] = parameters.detach().numpy()
    # print(parm)
