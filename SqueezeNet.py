import os
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms


class SqueezeNet(nn.Module):
    def __init__(self, embedding_dim=300):
        super(SqueezeNet, self).__init__()
        self.squeeze = models.squeezenet1_1(pretrained=True)
        self.squeeze.num_classes = embedding_dim
        final_conv = nn.Conv2d(512, self.squeeze.num_classes, kernel_size=1)
        self.squeeze.classifier[1] = final_conv
        self.linear = self.squeeze.classifier[1]
        # self.batch_norm = nn.BatchNorm1d(embedding_dim, momentum=0.01)
        self.init_weights()
    
    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)
    
    def forward(self, images):
        embed = self.squeeze(images)
        return embed
