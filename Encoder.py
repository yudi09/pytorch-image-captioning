import os
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms


class CNN(nn.Module):
    def __init__(self, architecture = 'alexnet', embedding_dim=512):
        super(CNN, self).__init__()
        self.architecture = architecture
        self.model, in_features = self.get_model()
        self.linear = nn.Linear(in_features, embedding_dim)
        # self.batch_norm = nn.BatchNorm1d(embedding_dim, momentum=0.01)
        self.init_weights()
    
    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)
    
    def forward(self, images):
        embed = self.model(images)
        embed = Variable(embed.data)
        embed = embed.view(embed.size(0), -1)
        embed = self.linear(embed)
        # embed = self.batch_norm(embed)
        return embed

    def get_model(self):
        if self.architecture == 'resnet18':
            return self.get_resnet18()
        elif self.architecture == 'alexnet':
            return self.get_alexnet()
    
    def get_resnet18(self):
        resnet18 = models.resnet18(pretrained=True)
        in_features = resnet18.fc.in_features
        modules = list(resnet18.children())[:-1]
        resnet18 = nn.Sequential(*modules)
        return resnet18, in_features

    def get_alexnet(self):
        alexnet = models.alexnet(pretrained=True)
        in_features = alexnet.classifier[6].in_features
        seq0 = list(alexnet.children())[0]
        seq1 = list(alexnet.children())[1][:-1]
        alexnet = nn.Sequential(*seq0, *seq1)
        # alexnet = nn.Sequential(*modules)
        return alexnet, in_features