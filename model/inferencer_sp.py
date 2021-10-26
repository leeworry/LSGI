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
from .GAT import SpGAT
from .layers import Classifier


class Inferencer(nn.Module):
    """ A sequence model for relation extraction. """

    def __init__(self, opt, emb_matrix=None):
        super(Inferencer, self).__init__()
        # self.encoder = RNNEncoder(opt, emb_matrix)
        self.gat = SpGAT(nfeat=opt['gat_input_dim'],
                       nhid=opt['gat_hid_dim'],
                       nclass=opt['num_class'],
                       dropout=opt['gat_dropout'],
                       nheads=opt['gat_heads'],
                       alpha=opt['gat_alpha'])
        # self.classifier = Classifier(opt)

    def forward(self, inputs, adj, pretrained=False):
        # encoding = self.encoder(inputs)
        logits = self.gat(inputs,adj)
        # logits2 = self.classifier(encoding)
        return logits, inputs

    def predict(self, inputs,adj):
        # encoding = self.encoder(inputs)
        logits = self.gat(inputs,adj)
        preds = F.softmax(logits, dim=-1)
        return preds

class Inferencer_tf(nn.Module):
    """ A sequence model for relation extraction. """

    def __init__(self, opt, emb_matrix=None):
        super(Inferencer_tf, self).__init__()
        self.encoder = RNNEncoder(opt, emb_matrix)
        self.gat = SpGAT(nfeat=opt['gat_input_dim'],
                       nhid=opt['gat_hid_dim'],
                       nclass=opt['num_class'],
                       dropout=opt['gat_dropout'],
                       nheads=opt['gat_heads'],
                       alpha=opt['gat_alpha'])
        # self.classifier = Classifier(opt['gat_output'],opt['num_class'])

    def forward(self, inputs, adj, pretrained=False):
        encoding = self.encoder(inputs)
        logits1 = self.gat(encoding,adj)
        # logits2 = self.classifier(encoding)
        return logits1, encoding

    def predict(self, inputs,adj):
        encoding = self.encoder(inputs)
        logits = self.gat(encoding,adj)
        preds = F.softmax(logits, dim=-1)
        return preds


class MultiGraphAttention(nn.Module):

    def __init__(self, d_features, attention_size, dropout=0.1):
        super(MultiGraphAttention, self).__init__()
        self.graph_attn = nn.Linear(attention_size,1,bias=False)
        self.w_q = nn.Linear(d_features, attention_size, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask=None):
        q = torch.tanh(self.w_q(inputs))
        vu = self.graph_attn(q).squeeze()
        alphas = torch.softmax(vu,dim=0)
        alphas = self.dropout(alphas).unsqueeze(-1)
        output = torch.sum(torch.mul(alphas, inputs),dim=0).squeeze()
        return output, alphas


class Inferencer_bi(nn.Module):
    """ A sequence model for relation extraction. """

    def __init__(self, opt):
        super(Inferencer_bi, self).__init__()
        # self.encoder = RNNEncoder(opt, emb_matrix)
        self.gat1 = SpGAT(nfeat=opt['gat_input_dim'],
                       nhid=opt['gat_hid_dim'],
                       nclass=opt['num_class'],
                       dropout=opt['gat_dropout'],
                       nheads=opt['gat_heads'],
                       alpha=opt['gat_alpha'])
        self.gat2 = SpGAT(nfeat=opt['gat_input_dim'],
                         nhid=opt['gat_hid_dim'],
                         nclass=opt['num_class'],
                         dropout=opt['gat_dropout'],
                         nheads=opt['gat_heads'],
                         alpha=opt['gat_alpha'])
        self.graph_attn = MultiGraphAttention(opt['gat_output'], opt['attn_dim'], dropout=0.1)
        self.classifier = Classifier(opt['gat_output'], opt['num_class'])

    def forward(self, inputs, adj, pretrained=False):
        # encoding = self.encoder(inputs)
        logits1 = self.gat1(inputs,adj[0])
        logits2 = self.gat2(inputs,adj[1])
        multi_logits = torch.stack([logits1, logits2], dim=0)
        output, attn = self.graph_attn(multi_logits)
        logits = F.log_softmax(self.classifier(output), dim=1)
        return logits, inputs

    def predict(self, inputs,adj):
        # encoding = self.encoder(inputs)
        logits1 = self.gat1(inputs, adj[0])
        logits2 = self.gat2(inputs, adj[1])
        multi_logits = torch.stack([logits1, logits2], dim=0)
        output, attn = self.graph_attn(multi_logits)
        logits = F.log_softmax(self.classifier(output), dim=1)
        preds = F.softmax(logits, dim=-1)
        return preds