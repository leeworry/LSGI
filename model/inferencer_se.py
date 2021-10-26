# -*- coding: utf-8 -*-
# @Time    : 2020/10/15 14:18
# @Author  : wanli.li
# @Email   : wanli.li@m.scnu.edu.cn
# @File    : inferencer.py
# @Software: PyCharm
import math
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
from model.GAT import SpGAT

class Inferencer(nn.Module):
    """ A sequence model for relation extraction. """

    def __init__(self, opt):
        super(Inferencer, self).__init__()
        self.opt = opt
        self.gat = SpGAT(nfeat=opt['gat_input_dim'],
                       nhid=opt['gat_hid_dim'],
                       nclass=opt['num_class'],
                       dropout=opt['gat_dropout'],
                       nheads=opt['gat_heads'],
                       alpha=opt['gat_alpha'])

    def forward(self, features, adj):
        logits = self.gat(features,adj)
        return logits


    def predict(self, features, adj):
        logits = self.gat(features, adj)
        preds = F.softmax(logits, dim=-1)
        return preds

    def get_features(self, features, adj):
        logits = self.gat.get_features(features, adj)
        return logits