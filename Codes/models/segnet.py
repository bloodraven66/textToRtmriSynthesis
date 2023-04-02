import torch
from torch import nn
from torch.nn import functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

#standard segnet model, with 3 decoders for 3 masks
class SegNet(nn.Module):
    def __init__(self,input_nbr=3,label_nbr=2):
        super(SegNet, self).__init__()

        batchNorm_momentum = 0.1

        self.conv11 = nn.Conv2d(input_nbr, 64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)

        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)

        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)

        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

        self.conv53d1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53d1 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv52d1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52d1 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv51d1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51d1 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

        self.conv43d1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d1 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv42d1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d1 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv41d1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d1 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)

        self.conv33d1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d1 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv32d1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d1 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv31d1 = nn.Conv2d(256,  128, kernel_size=3, padding=1)
        self.bn31d1 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)

        self.conv22d1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d1 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        self.conv21d1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d1 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)

        self.conv12d1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d1 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.conv11d1 = nn.Conv2d(64, label_nbr, kernel_size=3, padding=1)

        


        self.conv53d2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53d2 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv52d2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52d2 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv51d2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51d2 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

        self.conv43d2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d2 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv42d2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d2 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv41d2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d2 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)

        self.conv33d2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d2 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv32d2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d2 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv31d2 = nn.Conv2d(256,  128, kernel_size=3, padding=1)
        self.bn31d2 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)

        self.conv22d2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d2 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        self.conv21d2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d2 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)

        self.conv12d2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d2 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.conv11d2 = nn.Conv2d(64, label_nbr, kernel_size=3, padding=1)




        self.conv53d3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53d3 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv52d3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52d3 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv51d3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51d3 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

        self.conv43d3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d3 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv42d3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d3 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv41d3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d3 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)

        self.conv33d3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d3 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv32d3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d3 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv31d3 = nn.Conv2d(256,  128, kernel_size=3, padding=1)
        self.bn31d3 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)

        self.conv22d3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d3 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        self.conv21d3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d3 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)

        self.conv12d3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d3 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.conv11d3 = nn.Conv2d(64, label_nbr, kernel_size=3, padding=1)


    def forward(self, x):

        # Stage 1
        x11 = F.relu(self.bn11(self.conv11(x)))
        x12 = F.relu(self.bn12(self.conv12(x11)))
        x1p, id1 = F.max_pool2d(x12,kernel_size=2, stride=2,return_indices=True)

        # Stage 2
        x21 = F.relu(self.bn21(self.conv21(x1p)))
        x22 = F.relu(self.bn22(self.conv22(x21)))
        x2p, id2 = F.max_pool2d(x22,kernel_size=2, stride=2,return_indices=True)

        # Stage 3
        x31 = F.relu(self.bn31(self.conv31(x2p)))
        x32 = F.relu(self.bn32(self.conv32(x31)))
        x33 = F.relu(self.bn33(self.conv33(x32)))
        x3p, id3 = F.max_pool2d(x33,kernel_size=2, stride=2,return_indices=True)

        # Stage 4
        x41 = F.relu(self.bn41(self.conv41(x3p)))
        x42 = F.relu(self.bn42(self.conv42(x41)))
        x43 = F.relu(self.bn43(self.conv43(x42)))
        x4p, id4 = F.max_pool2d(x43,kernel_size=2, stride=2,return_indices=True)

        # Stage 5
        x51 = F.relu(self.bn51(self.conv51(x4p)))
        x52 = F.relu(self.bn52(self.conv52(x51)))
        x53 = F.relu(self.bn53(self.conv53(x52)))
        x5p, id5 = F.max_pool2d(x53,kernel_size=2, stride=2,return_indices=True)


        # Stage 5d
        x5d1 = F.max_unpool2d(x5p, id5, kernel_size=2, stride=2)
        x53d1 = F.relu(self.bn53d1(self.conv53d1(x5d1)))
        x52d1 = F.relu(self.bn52d1(self.conv52d1(x53d1)))
        x51d1 = F.relu(self.bn51d1(self.conv51d1(x52d1)))

        # Stage 4d
        x4d1 = F.max_unpool2d(x51d1, id4, kernel_size=2, stride=2)
        x43d1 = F.relu(self.bn43d1(self.conv43d1(x4d1)))
        x42d1 = F.relu(self.bn42d1(self.conv42d1(x43d1)))
        x41d1 = F.relu(self.bn41d1(self.conv41d1(x42d1)))

        # Stage 3d
        x3d1 = F.max_unpool2d(x41d1, id3, kernel_size=2, stride=2)
        x33d1 = F.relu(self.bn33d1(self.conv33d1(x3d1)))
        x32d1 = F.relu(self.bn32d1(self.conv32d1(x33d1)))
        x31d1 = F.relu(self.bn31d1(self.conv31d1(x32d1)))

        # Stage 2d
        x2d1 = F.max_unpool2d(x31d1, id2, kernel_size=2, stride=2)
        x22d1 = F.relu(self.bn22d1(self.conv22d1(x2d1)))
        x21d1 = F.relu(self.bn21d1(self.conv21d1(x22d1)))

        # Stage 1d
        x1d1 = F.max_unpool2d(x21d1, id1, kernel_size=2, stride=2)
        x12d1 = F.relu(self.bn12d1(self.conv12d1(x1d1)))
        x11d1 = self.conv11d1(x12d1)




        x5d2 = F.max_unpool2d(x5p, id5, kernel_size=2, stride=2)
        x53d2 = F.relu(self.bn53d2(self.conv53d2(x5d2)))
        x52d2 = F.relu(self.bn52d2(self.conv52d2(x53d2)))
        x51d2 = F.relu(self.bn51d2(self.conv51d2(x52d2)))

        # Stage 4d
        x4d2 = F.max_unpool2d(x51d2, id4, kernel_size=2, stride=2)
        x43d2 = F.relu(self.bn43d2(self.conv43d2(x4d2)))
        x42d2 = F.relu(self.bn42d2(self.conv42d2(x43d2)))
        x41d2 = F.relu(self.bn41d2(self.conv41d2(x42d2)))

        # Stage 3d
        x3d2 = F.max_unpool2d(x41d2, id3, kernel_size=2, stride=2)
        x33d2 = F.relu(self.bn33d2(self.conv33d2(x3d2)))
        x32d2 = F.relu(self.bn32d2(self.conv32d2(x33d2)))
        x31d2 = F.relu(self.bn31d2(self.conv31d2(x32d2)))

        # Stage 2d
        x2d2 = F.max_unpool2d(x31d2, id2, kernel_size=2, stride=2)
        x22d2 = F.relu(self.bn22d2(self.conv22d2(x2d2)))
        x21d2 = F.relu(self.bn21d2(self.conv21d2(x22d2)))

        # Stage 1d
        x1d2 = F.max_unpool2d(x21d2, id1, kernel_size=2, stride=2)
        x12d2 = F.relu(self.bn12d2(self.conv12d2(x1d2)))
        x11d2 = self.conv11d2(x12d2)




        x5d3 = F.max_unpool2d(x5p, id5, kernel_size=2, stride=2)
        x53d3 = F.relu(self.bn53d3(self.conv53d3(x5d3)))
        x52d3 = F.relu(self.bn52d3(self.conv52d3(x53d3)))
        x51d3 = F.relu(self.bn51d3(self.conv51d3(x52d3)))

        # Stage 4d
        x4d3 = F.max_unpool2d(x51d3, id4, kernel_size=2, stride=2)
        x43d3 = F.relu(self.bn43d3(self.conv43d3(x4d3)))
        x42d3 = F.relu(self.bn42d3(self.conv42d3(x43d3)))
        x41d3 = F.relu(self.bn41d3(self.conv41d3(x42d3)))

        # Stage 3d
        x3d3 = F.max_unpool2d(x41d3, id3, kernel_size=2, stride=2)
        x33d3 = F.relu(self.bn33d3(self.conv33d3(x3d3)))
        x32d3 = F.relu(self.bn32d3(self.conv32d3(x33d3)))
        x31d3 = F.relu(self.bn31d3(self.conv31d3(x32d3)))

        # Stage 2d
        x2d3 = F.max_unpool2d(x31d3, id2, kernel_size=2, stride=2)
        x22d3 = F.relu(self.bn22d3(self.conv22d3(x2d3)))
        x21d3 = F.relu(self.bn21d3(self.conv21d3(x22d3)))

        # Stage 1d
        x1d3 = F.max_unpool2d(x21d3, id1, kernel_size=2, stride=2)
        x12d3 = F.relu(self.bn12d3(self.conv12d3(x1d3)))
        x11d3 = self.conv11d3(x12d3)

        return x11d1, x11d2, x11d3

