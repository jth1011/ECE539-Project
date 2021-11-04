import torch
import torch.nn as nn

from network_blocks import ConvBlock, DeconvBlock, UpBlock, DownBlock

class Net(nn.Module):

    def __init__(self, channels, features, filt):
        super(Net, self).__init__()

        self.lay0 = ConvBlock(channels, features, 3, 1, 1, activ='prelu')
        self.lay1 = ConvBlock(features, filt, 1, 1, 0, activ='prelu')
        self.lay2 = ConvBlock(filt, filt, 1, 1, 0, activ='prelu')

        kernel = 4
        stride = 2
        padding = 2

        self.up1 = UpBlock(filt, kernel, stride, padding)
        self.down1 = DownBlock(filt, kernel, stride, padding)
        # self.cnn_conv = nn.sequential(
        #     nn.conv2d(channels, 16, kernel_size=5, stride=1, padding=1),
        #     nn.batchnorm2d(16),
        #     nn.prelu(true),
        #     nn.conv2d(16, 64, kernel_size=3, stride=1, padding=1),
        #     nn.batchnorm2d(64),
        #     nn.relu(true),
        #     nn.conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        #     nn.batchnorm2d(64),
        #     nn.relu(true),
        #     nn.conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        #     nn.batchnorm2d(128),
        #     nn.relu(true)
        # )

        # self.cnn_deconv = nn.Sequential(
        #     nn.ConvTranspose2d(128, 16, kernel_size=3, stride=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2),
        # )



    def forward(self, x):
        x = self.lay0(x)
        x = self.lay1(x)
        x = self.lay2(x)

        
        x = self.cnn_conv(x)
        x = self.cnn_deconv(x)
        return x
