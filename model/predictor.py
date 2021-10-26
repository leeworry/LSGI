import math
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F

from .layers import Classifier
from .encoder import RNNEncoder, CNNEncoder


class Predictor(nn.Module):
    """ A sequence model for relation extraction. """

    def __init__(self, opt, emb_matrix=None):
        super(Predictor, self).__init__()
        self.encoder = RNNEncoder(opt, emb_matrix)
        self.classifier = Classifier(opt['hidden_dim'], opt['num_class'])

    def forward(self, inputs, direction=None):
        if direction is not None:
            encoding = self.encoder(inputs, direction=direction)
        else:
            encoding = self.encoder(inputs, direction=None)
        logits = self.classifier(encoding)
        return logits, encoding

    def predict(self, inputs):
        encoding = self.encoder(inputs)
        logits = self.classifier(encoding)
        preds = F.softmax(logits, dim=-1)
        return preds