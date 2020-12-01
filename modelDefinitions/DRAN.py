import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary
from modelDefinitions.basicBlocks import *    



class DynamicResAttNet(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DynamicResAttNet, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        self.inputLayer = nn.Sequential(*layers)

        block1 = []

        for _ in range(5):
            block1.append(Dynamic_conv2d(in_planes=features, out_planes=features, kernel_size=kernel_size, padding=padding))
            block1.append(nn.BatchNorm2d(features))
            block1.append(nn.ReLU(inplace=True))
        self.block1 = nn.Sequential(*block1)
        self.noiseGate1 = GatedConv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1)
       
        block2 = []

        for _ in range(5):
            block2.append(Dynamic_conv2d(in_planes=features, out_planes=features, kernel_size=kernel_size, padding=padding))
            block2.append(nn.BatchNorm2d(features))
            block2.append(nn.ReLU(inplace=True))
        self.block2 = nn.Sequential(*block2)
        self.noiseGate2 = GatedConv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1)


        block3 = []

        for _ in range(5):
            block3.append(Dynamic_conv2d(in_planes=features, out_planes=features, kernel_size=kernel_size, padding=padding))
            block3.append(nn.BatchNorm2d(features))
            block3.append(nn.ReLU(inplace=True))
        self.block3 = nn.Sequential(*block3)
        self.noiseGate3 = GatedConv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1)

        self.outputLayer = nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False)
    
    def forward(self, x):
        x0 = self.inputLayer(x)

        x1 = self.block1(x0) + self.noiseGate1(x0)
        x2 = self.block1(x1) + self.noiseGate2(x1)
        x3 = self.block1(x2) + self.noiseGate3(x2)

        return torch.tanh(self.outputLayer(x3))


class DynamicDnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DynamicDnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(Dynamic_conv2d(in_planes=features, out_planes=features, kernel_size=kernel_size, padding=padding))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        out = self.dncnn(x)
        return torch.tanh(out)
#net = DynamicResNet(3)
#summary(net, input_size = (3, 128, 128))
#print ("reconstruction network")