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
        #self.dropout4 = nn.Dropout(0.3)
        #self.pool4=nn.MaxPool2d(2,2)
        #self.conv5 = nn.Conv2d(in_channels=80, out_channels=96, kernel_size=3,bias=False)
        #self.conv5_bn = nn.BatchNorm2d(96)
        #self.dropout5 = nn.Dropout(0.3)
        #self.conv6 = nn.Conv2d(in_channels=96, out_channels=112, kernel_size=3, bias=False)
        #self.conv6_bn = nn.BatchNorm2d(112)
        #self.dropout6 = nn.Dropout(0.3)
        #self.conv7 = nn.Conv2d(in_channels=112, out_channels=128, kernel_size=3,bias=False)
        #self.conv7_bn = nn.BatchNorm2d(128)
        #self.dropout7 = nn.Dropout(0.3)
        #self.conv8 = nn.Conv2d(in_channels=128, out_channels=144, kernel_size=3, bias=False)
        #self.conv8_bn = nn.BatchNorm2d(144)
        #self.pool5=nn.MaxPool2d(2,2)
        #self.dropout8 = nn.Dropout(0.3)
        #self.conv9 = nn.Conv2d(in_channels=144, out_channels=160, kernel_size=3, bias=False)
        #self.conv9_bn = nn.BatchNorm2d(160)
        #self.dropout9 = nn.Dropout(0.3)
        #self.conv10 = nn.Conv2d(in_channels=160, out_channels=176, kernel_size=3, bias=False)
        #self.conv10_bn = nn.BatchNorm2d(176)
        #self.dropout10 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(in_features=3072, out_features=10,bias=False) #11264
        self.fc1_bn = nn.BatchNorm1d(10)
        #self.relu=nn.ReLU()
    def get_logits(self, x):
        conv1 = F.relu(self.conv1_bn(self.conv1(x)))
        conv2 = F.relu(self.conv2_bn(self.conv2(conv1)))
        conv3 = F.relu(self.conv3_bn(self.conv3(conv2)))
        conv4 = F.relu(self.conv4_bn(self.conv4(conv3)))
        #conv5 = F.relu(self.conv5_bn(self.conv5(conv4)))
        #conv6 = F.relu(self.conv6_bn(self.conv6(conv5)))
        #conv7 = F.relu(self.conv7_bn(self.conv7(conv6)))
        #conv8 = F.relu(self.conv8_bn(self.conv8(conv7)))
        #conv9 = F.relu(self.conv9_bn(self.conv9(conv8)))
        #conv10 = F.relu(self.conv10_bn(self.conv10(conv9)))

        flat1 = torch.flatten(conv4, 1)
        logits = self.fc1_bn(self.fc1(flat1))     #pish bini kham az laye final cnn nn ke b softmax midim
        return logits
    def forward(self, x):
        logits = self.get_logits(x)
        return logits #F.log_softmax(logits, dim=1)
