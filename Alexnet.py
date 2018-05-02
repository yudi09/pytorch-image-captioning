import os
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms


class Alexnet(nn.Module):
    def __init__(self, embedding_dim=512):
        super(Alexnet, self).__init__()
        self.alexnet = models.alexnet(pretrained=True)
        in_features = self.alexnet.classifier[6].in_features
        self.linear = nn.Linear(in_features, embedding_dim)
        self.alexnet.classifier[6] = self.linear
        # self.batch_norm = nn.BatchNorm1d(embedding_dim, momentum=0.01)
        self.init_weights()
    
    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)
    
    def forward(self, images):
        embed = self.alexnet(images)
        # embed = Variable(embed.data)
        # embed = embed.view(embed.size(0), -1)
        # embed = self.linear(embed)
        # embed = self.batch_norm(embed)
        return embed
