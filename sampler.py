# -*- coding: utf-8 -*-
"""
@Time ： 2021/1/22 17:00
@Auth ： PUCHAZHONG
@File ：sampler.py
@IDE ：PyCharm
@CONTACT: zhonghw@zhejianglab.com
"""


from collections import defaultdict
import numpy as np
import copy
import random

from torch.utils.data.sampler import Sampler
class RandomDomainSampler(Sampler):
    def __init__(self, data_source, batch_size, domain_label):

        self.data_source = data_source
        self.batch_size = batch_size
        self.domain_label = list(domain_label.cpu().item())
        self.length = len(domain_label)

    def __iter__(self):
        used_idx = []
        sou_idx = list(np.where(self.domain_label == 1.))
        tar_idx = list(np.where(self.domain_label == 0.))
        if len(sou_idx) >= self.batch_size // 2:
            temp = np.random.choice(sou_idx, self.batch_size // 2)




        return iter(final_idxs)

    def __len__(self):
        return self.length