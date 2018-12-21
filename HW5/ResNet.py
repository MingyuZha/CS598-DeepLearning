import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from time import *
from copy import *


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResidualBlock, self).__init__()
        self.weight_layers = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1),
            # nn.BatchNorm2d(num_features=out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            # nn.BatchNorm2d(out_channel)
            )
        self.short_cut = nn.Sequential()
        if (stride != 1 or in_channel != out_channel):
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, stride, bias=False),
                # nn.BatchNorm2d(out_channel)
                )
    def forward(self, x):
        out = self.weight_layers(x)
        out += self.short_cut(x)
        out = nn.functional.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, D):
        super(ResNet, self).__init__()
        #define basic block
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout()
            )

        self.blocks1 = self.make_layer(block=ResidualBlock, in_channel=32, out_channel=32, num_basic_blocks=2, stride=1)
        self.blocks2 = self.make_layer(ResidualBlock, 32, 64, 4, 2)
        self.blocks3 = self.make_layer(ResidualBlock, 64, 128, 4, 2)
        self.blocks4 = self.make_layer(ResidualBlock, 128, 256, 2, 2)

        self.MaxPool = nn.MaxPool2d(3,1)

        self.fc2 = nn.Sequential(
            nn.Linear(256*2*2, D),
            # nn.BatchNorm1d(D) 
            # nn.Softmax(dim=1)
            )
        
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = self.blocks4(x)
        x = self.MaxPool(x)
        x = x.view(-1, 256*2*2)
        # x = self.fc1(x)
        x = self.fc2(x)
        return x


    def make_layer(self, block, in_channel, out_channel, num_basic_blocks, stride):
        layers = []
        for i in range(num_basic_blocks):
            if (i == 0): 
                s = stride
                exact_inchannel = in_channel
            else: 
                s = 1
                exact_inchannel = out_channel
            layers.append(block(exact_inchannel, out_channel, s))
        return nn.Sequential(*layers)







