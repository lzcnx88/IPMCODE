from __future__ import print_function
import scipy.io as sio
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pooling_and_full_propagate import *


class PPNet(nn.Module):
    def __init__(self, nn_word_vectors, none_nn_word_vectors, similarity):
        super(PPNet, self).__init__()
        nn_dist = euclidean_dist(nn_word_vectors, nn_word_vectors)
        nn_dist = 1.0 / (nn_dist + 0.001)
        nn_dist = nn_dist - torch.diag(nn_dist.diag())
        nn_dist = nn_dist / torch.sum(nn_dist, 1)
        self.s1 = nn_dist

        nn_dist = euclidean_dist(none_nn_word_vectors, none_nn_word_vectors)
        nn_dist = 1.0 / (nn_dist + 0.001)
        nn_dist = nn_dist - torch.diag(nn_dist.diag())
        nn_dist = nn_dist / torch.sum(nn_dist, 1)
        self.s2 = nn_dist

        # self.k1 = 10
        # self.k2 = 10

        # self.nn_pooling = Pooling(nn_word_vectors, self.k1)
        # self.none_nn_pooling = Pooling(none_nn_word_vectors, self.k2)
        self.propagate1 = Propagate(0.5)
        self.propagate2 = Propagate(0.5)

        self.similarity = similarity

        self.conv1_w = nn.Conv2d(1, 32, kernel_size=(1, 5))
        self.conv1_h = nn.Conv2d(32, 32, kernel_size=(3, 1))
        self.conv2_w = nn.Conv2d(32, 64, kernel_size=(1, 3))
        self.conv2_h = nn.Conv2d(64, 64, kernel_size=(3, 1))
        self.conv3_w = nn.Conv2d(64, 128, kernel_size=(1, 3))
        self.conv3_h = nn.Conv2d(128, 128, kernel_size=(3, 1))
        self.conv4_w = nn.Conv2d(128, 256, kernel_size=(1, 3))
        self.conv4_h = nn.Conv2d(256, 256, kernel_size=(3, 1))
        self.conv5_w = nn.Conv2d(256, 512, kernel_size=(1, 3))
        self.conv5_h = nn.Conv2d(512, 512, kernel_size=(3, 1))
        self.conv2_drop = nn.Dropout2d()
        self.bn1 = nn.BatchNorm2d(32)
        self.bn11 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn22 = nn.BatchNorm2d(64)
        # self.fc1 = nn.Linear(186624, 512)
        self.fc1 = nn.Linear(691456, 512)
        self.fc2 = nn.Linear(512, 283)
        # self.fc2 = nn.Linear(512, 129)

        self.weight_init(self.conv1_w)
        self.weight_init(self.conv1_h)
        self.weight_init(self.conv2_h)
        self.weight_init(self.conv2_w)
        self.cnt = 0

    def set_test(self):
        self.training = False

    def set_path(self, path):
        self.path = path

    def log_(self, x, bt=2):
        x = F.normalize(x)
        x_size = x.size()
        log_bt = torch.zeros(x_size).cuda()
        log_bt[:] = bt
        res = x.float().log1p() / log_bt.float().log()
        return res

    def norm(self, x):
        m = x.view(x.size(0), -1)
        m_sum = torch.sum(m, 1).view(x.size(0), 1) + 1e-7
        m = m / m_sum
        m = m.view(x.size(0), 1, -1)
        return m

    def bilinear_size(self, in_size, k):
        p = torch.floor(0.5 * ((0.5 + torch.ceil(torch.Tensor([in_size * 1.0 / k]))) * k - in_size))
        out_size = torch.floor((in_size + 2 * p - k) / k + 1)

        return out_size

    def weight_init(self, m):
        if (isinstance(m, nn.Conv2d)):
            print('conv layer initial ...')
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))

    def forward(self, data):
        x = data
        x = F.max_pool2d(self.conv1_w(x), (1, 2))
        x = self.bn11(x)
        x = F.relu(x)
        x = F.max_pool2d(self.conv1_h(x), (2, 1))
        x = self.bn1(x)
        x = F.relu(x)

        x = F.max_pool2d(self.conv2_w(x), (1, 2))
        x = self.bn22(x)
        x = F.relu(x)
        x = F.max_pool2d(self.conv2_h(x), (2, 1))
        x = self.bn2(x)
        x = F.relu(x)

        # x = F.relu(F.max_pool2d(self.conv4(x), 2))
        # x = F.relu(F.max_pool2d(self.conv5(x), 2))
        # if(self.training == False):
        #     save_feature_to_img(x, 'visdf/1_'+self.path, 10)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.8, training=self.training)
        x = self.fc2(x)

        # x1 = x1.view(-1, x1.size(2))
        # x2 = x2.view(-1, x2.size(2))

        # product = self.mcb(x1, x2)

        return F.log_softmax(x, dim=1)


if (__name__ == '__main__'):
    wv1 = torch.rand(2048, 512)
    wv2 = torch.rand(2048, 512)
    s = torch.randn(2048, 2048)
    ppnet = PPNet(s)

    x1 = torch.rand(10, 1, 2048).cuda()
    x2 = torch.rand(10, 1, 2048).cuda()

    res = ppnet(x1, x2)
    print(ppnet)




