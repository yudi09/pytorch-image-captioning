import os
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms


class DenseNet(nn.Module):
    def __init__(self, embedding_dim=300):
        super(DenseNet, self).__init__()
        self.dense = models.densenet121(pretrained=True)
        self.linear = nn.Linear(self.dense.classifier.in_features, embedding_dim)
        self.dense.classifier = self.linear
        # self.batch_norm = nn.BatchNorm1d(embedding_dim, momentum=0.01)
        self.init_weights()
    
    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)
    
    def forward(self, images):
        embed = self.dense(images)
        return embed
