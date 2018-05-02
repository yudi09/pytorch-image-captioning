import os
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms


class Inception(nn.Module):
    def __init__(self, embedding_dim=300):
        super(Inception, self).__init__()
        self.inception = models.inception_v3(pretrained=True)
        in_features = self.inception.fc.in_features
        self.linear = nn.Linear(in_features, embedding_dim)
        self.inception.fc = self.linear
        # self.batch_norm = nn.BatchNorm1d(embedding_dim, momentum=0.01)
        self.init_weights()
    
    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)
    
    def forward(self, images):
        embed = self.inception(images)
        return embed
