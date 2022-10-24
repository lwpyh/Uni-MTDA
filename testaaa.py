# from torchvision import transforms
# from torchvision.datasets import CIFAR10
# import torch
# import torch.nn as nn
# import torch.optim as optim
#
#
#
# def get_dataset(cls):
#     MEAN = [0.49139968, 0.48215827, 0.44653124]
#     STD = [0.24703233, 0.24348505, 0.26158768]
#     transf = [
#         transforms.Resize((64,64)),
#         transforms.RandomCrop(64, padding=4),
#         transforms.RandomHorizontalFlip()
#     ]
#     normalize = [
#         transforms.ToTensor(),
#         transforms.Normalize(MEAN, STD)
#     ]
#
#     train_transform = transforms.Compose(transf + normalize)
#     valid_transform = transforms.Compose([transforms.Resize((64,64))] + normalize)
#
#     if cls == "cifar10":
#         dataset_train = CIFAR10(root="./data", train=True, download=True, transform=train_transform)
#         dataset_valid = CIFAR10(root="./data", train=False, download=True, transform=valid_transform)
#     else:
#         raise NotImplementedError
#     return dataset_train, dataset_valid
#
#
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1_1 = nn.Conv2d(in_channels = 3, out_channels = 10, kernel_size = 3)
#         self.relu1_1 = nn.ReLU()
#         self.conv1_2 = nn.Conv2d(in_channels = 10, out_channels = 10, kernel_size = 3)
#         self.relu1_2 = nn.ReLU()
#         self.max_pool_1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
#         self.conv2_1 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3)
#         self.relu2_1 = nn.ReLU()
#         self.conv2_2 = nn.Conv2d(in_channels = 10, out_channels = 10, kernel_size = 3)
#         self.relu2_2 = nn.ReLU()
#         self.max_pool_2 = nn.MaxPool2d(kernel_size = 2, stride=2)
#         self.fc1 = nn.Linear(1690,256)
#         self.fc2 = nn.Linear(256,256)
#         self.fc3 = nn.Linear(256,10)
#
#
#
#     def forward(self, x):
#         x = self.conv1_1(x)
#         x = self.relu1_1(x)
#         x = self.conv1_2(x)
#         x = self.relu1_2(x)
#         x = self.max_pool_1(x)
#         x = self.conv2_1(x)
#         x = self.relu2_1(x)
#         x = self.conv2_2(x)
#         x = self.relu2_2(x)
#         x = self.max_pool_2(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         y = self.fc3(x)
#         return y
#
#
# def train(epoch, net, trainloader):
#
#     net.train()
#     train_loss = 0
#     correct = 0
#     total = 0
#     for batch_idx, (inputs, targets) in enumerate(trainloader):
#         inputs, targets = inputs.cuda(), targets.cuda()
#         optimizer.zero_grad()
#         outputs = net(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()
#
#         train_loss += loss.item()
#         _, predicted = outputs.max(1)
#         total += targets.size(0)
#         correct += predicted.eq(targets).sum().item()
#     print('\nEpoch: {:03d}, Loss: {:.4f}'.format(epoch, train_loss / (batch_idx+1)))
#
#
#
# def test(epoch, net, testloader):
#     net.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(testloader):
#             inputs, targets = inputs.cuda(), targets.cuda()
#             outputs = net(inputs)
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()
#     print('\nEpoch: {:03d}, test_acc: {:.4f}'.format(epoch, correct/total))
#
# if __name__ =="__main__":
#     net = CNN().cuda()
#     print(net)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(net.parameters(), lr=0.1,
#                           momentum=0.9, weight_decay=5e-4)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
#     dataset_train, dataset_valid = get_dataset("cifar10")
#     train_loader = torch.utils.data.DataLoader(dataset_train,
#                                                     batch_size=128,
#                                                     num_workers=4)
#     test_loader = torch.utils.data.DataLoader(dataset_valid,
#                                                     batch_size=128,
#                                                     num_workers=4)
#     for epoch in range(0, 200):
#         train(epoch,net,train_loader)
#         scheduler.step()
#         if (epoch + 1) % 10 == 0:
#             test(epoch,net,test_loader)
#
import numpy as np
# a = np.array([  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
#    0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
#    0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
#    0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
#    0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
#    0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
#    0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
#    0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
#    0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
#    0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
#    0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
#    0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
#    0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
#    0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
#    0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
#    0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
#    0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
#    0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
#    0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
#    0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
#    0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
#    0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
#    0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   1.,   1.,   1.,   1.,
#    1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,
#    1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,
#    1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,
#    1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,
#    1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,
#    1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,
#    1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,
#    1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,
#    1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,
#    1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,
#    1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,
#    1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,
#    1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,
#    1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,
#    1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   2.,   2.,   2.,   2.,   2.,
#    2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,
#    2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,
#    2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,
#    2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,
#    2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,
#    2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,
#    2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,
#    2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   3.,   3.,   3.,   3.,   3.,   3.,
#    3.,   3.,   3.,   3.,   3.,   3.,   3.,   3.,   3.,   3.,   3.,   3.,   3.,   3.,
#    3.,   3.,   3.,   3.,   3.,   3.,   3.,   3.,   3.,   3.,   3.,   3.,   3.,   3.,
#    3.,   3.,   3.,   3.,   3.,   3.,   3.,   3.,   3.,   3.,   3.,   3.,   3.,   3.,
#    3.,   3.,   3.,   3.,   3.,   3.,   3.,   3.,   3.,   3.,   3.,   3.,   3.,   3.,
#    3.,   3.,   3.,   3.,   3.,   3.,   3.,   3.,   3.,   3.,   3.,   3.,   3.,   3.,
#    3.,   3.,   3.,   3.,   3.,   3.,   3.,   3.,   3.,   3.,   4.,   4.,   4.,   4.,
#    4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,
#    4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,
#    4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,
#    4.,   4.,   5.,   5.,   5.,   5.,   5.,   5.,   5.,   5.,   5.,   5.,   5.,   5.,
#    5.,   5.,   5.,   5.,   5.,   5.,   5.,   5.,   5.,   5.,   5.,   5.,   5.,   5.,
#    5.,   5.,   5.,   5.,   5.,   5.,   5.,   6.,   6.,   6.,   6.,   6.,   6.,   6.,
#    6.,   6.,   6.,   6.,   6.,   6.,   6.,   6.,   6.,   6.,   6.,   6.,   6.,   6.,
#    6.,   7.,   7.,   7.,   7.,   7.,   7.,   7.,   7.,   7.,   7.,   7.,   7.,   7.,
#    7.,   7.,   7.,   7.,   7.,   7.,   7.,   7.,   7.,   8.,   8.,   8.,   8.,   8.,
#    8.,   8.,   8.,   8.,   8.,   8.,   9.,   9.,   9.,   9.,   9.,   9.,   9.,   9.,
#    9.,   9.,   9.,   9.,   9.,   9.,  10.,  10.,  10.,  10.,  10.,  11.,  11.,  11.,
#   11.,  12.,  12.,  12.,  12.,  13.,  13.,  13.,  13.,  14.,  14.,  15.,  15.,  15.,
#   15.,  15.,  16.,  16.,  17.,  17.,  18.,  19.,  20.,  20.,  21.,  24.,  24.,  25.,
#   27.,  28.,  31.,  31.,  31.,  36.,  36.,  37.,  39.,  39.,  45.,  45.,  49.,  49.,
#   50.,  52.,  55.,  56.,  57.,  59.,  59.,  60.,  61.,  63.,  63.,  64.,  67.,  67.,
#   68.,  69.,  71.,  72.,  72.,  72.,  73.,  74.,  74.,  74.,  76.,  77.,  77.,  79.,
#   80.,  80.,  82.,  82.,  82.,  83.,  84.,  85.,  85.,  86.,  86.,  86.,  87.,  88.,
#   89.,  89.,  90.,  90.,  91.,  91.,  92.,  92.,  92.,  93.,  94.,  94.,  95.,  95.,
#   95.,  95.,  96.,  97.,  98., 100., 101., 101., 102., 103., 104., 105., 107., 111.,
#  116., 138., 140., 140., 176., 205.,])
a = np.array([1.7938, 0.5614, 1.2049, 1.0623, 3.0417, 0.8227, 1.3623, 2.1299, 1.1429,
        1.2251, 1.6242, 1.5362, 1.9489, 1.5559, 1.0905, 1.1481, 1.0587, 1.0048,
        2.4017, 1.5015, 1.7167, 0.6038, 1.5482, 2.9001, 1.2387, 0.8205, 1.1434,
        0.4271, 0.6346, 0.6452, 0.6087, 1.2500, 0.9461, 0.3575, 0.9926, 1.0025,
        0.6549, 0.4094, 0.5405, 0.4147, 0.8827, 0.9662, 0.3679, 0.5707, 0.8617,
        0.9523, 0.6608, 0.4182, 0.7362, 0.3478, 0.6812, 0.5528, 0.4729, 0.4440,
        0.4850, 1.0421, 0.5124, 0.5493, 0.7459, 1.3507, 0.4857, 0.7721, 0.3976,
        1.2517, 0.4195])
# a = np.array(a)

suitable_th = 0
max_g = 0
for i in a:
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
print(suitable_th)
print(np.sum(a>=suitable_th))


t0 = np.mean(a)

def iter_thresh(a,t):
    x1 = np.mean(a[a>t])
    x2 = np.mean(a[a<=t])
    return (x1+x2)/2

while iter_thresh(a,t0) != t0:
    t0 = iter_thresh(a,t0)

print(t0)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2).fit(a.reshape(-1,1))
thresh = np.sum(a>max(a[np.where(kmeans.labels_ == 0)]))
print(thresh)
