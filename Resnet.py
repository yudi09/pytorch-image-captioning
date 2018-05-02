import os
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms


class Resnet(nn.Module):
    def __init__(self, embedding_dim=256):
        super(Resnet, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        in_features = self.resnet18.fc.in_features
        modules = list(self.resnet18.children())[:-1]
        self.resnet18 = nn.Sequential(*modules)
        self.linear = nn.Linear(in_features, embedding_dim)
        # self.batch_norm = nn.BatchNorm1d(embedding_dim, momentum=0.01)
        self.init_weights()
    
    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)
    
    def forward(self, images):
        embed = self.resnet18(images)
        embed = Variable(embed.data)
        embed = embed.view(embed.size(0), -1)
        embed = self.linear(embed)
        # embed = self.batch_norm(embed)
        return embed
