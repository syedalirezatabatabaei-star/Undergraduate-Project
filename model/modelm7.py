import torch    
from torch import nn
import torch.nn.functional as F



class CNN1M7 (nn.Module):
    def __init__(self):
        super( CNN1M7 , self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=48, kernel_size=7, bias=False)
        self.conv1_bn = nn.BatchNorm2d(48)
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=7,bias=False)
        self.conv2_bn = nn.BatchNorm2d(96)
        self.conv3 = nn.Conv2d(in_channels=96, out_channels=144, kernel_size=7, bias=False)
        self.conv3_bn = nn.BatchNorm2d(144)
        self.conv4 = nn.Conv2d(in_channels=144, out_channels=192, kernel_size=7, bias=False)
        self.conv4_bn = nn.BatchNorm2d(192)
        self.fc1 = nn.Linear(in_features=3072, out_features=10,bias=False) #11264
        self.fc1_bn = nn.BatchNorm1d(10)
    def get_logits(self, x):
        conv1 = F.relu(self.conv1_bn(self.conv1(x)))
        conv2 = F.relu(self.conv2_bn(self.conv2(conv1)))
        conv3 = F.relu(self.conv3_bn(self.conv3(conv2)))
        conv4 = F.relu(self.conv4_bn(self.conv4(conv3)))

        flat1 = torch.flatten(conv4, 1)
        logits = self.fc1_bn(self.fc1(flat1))     #pish bini kham az laye final cnn nn ke b softmax midim
        return logits
    def forward(self, x):
        logits = self.get_logits(x)
        return logits #F.log_softmax(logits, dim=1)
