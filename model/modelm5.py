import torch
from torch import nn
import torch.nn.functional as F


class CNN1M5 (nn.Module):
    def __init__(self):
        super( CNN1M5 , self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, bias=False)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5,bias=False)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=5, bias=False)
        self.conv3_bn = nn.BatchNorm2d(96)
        self.conv4 = nn.Conv2d(in_channels=96, out_channels=128, kernel_size=5, bias=False)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=160, kernel_size=5,bias=False)
        self.conv5_bn = nn.BatchNorm2d(160)
        self.fc1 = nn.Linear(in_features=10240, out_features=10,bias=False) #11264
        self.fc1_bn = nn.BatchNorm1d(10)
    def get_logits(self, x):
        conv1 = F.relu(self.conv1_bn(self.conv1(x)))
        conv2 = F.relu(self.conv2_bn(self.conv2(conv1)))
        conv3 = F.relu(self.conv3_bn(self.conv3(conv2)))
        conv4 = F.relu(self.conv4_bn(self.conv4(conv3)))
        conv5 = F.relu(self.conv5_bn(self.conv5(conv4)))

        flat1 = torch.flatten(conv5, 1)
        logits = self.fc1_bn(self.fc1(flat1))     #pish bini kham az laye final cnn nn ke b softmax midim
        return logits
    def forward(self, x):
        logits = self.get_logits(x)
        return logits #F.log_softmax(logits, dim=1)
