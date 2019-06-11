import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math


class SpecgramNet(nn.Module):
    """Empirically created small model roughly optimized to handle spectrogram images.
    """

    def __init__(self, num_classes=1000):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7,3), stride=(2,1),
                      padding=1, dilation=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5,3), stride=1,
                      padding=1, dilation=1),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,5), stride=(1,2),
                      padding=1, dilation=1),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2,
                      padding=1, dilation=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.PReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x

def specgram_net(pretrained=False, **kwargs):
    return SpecgramNet(**kwargs)
