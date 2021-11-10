import torch
import torch.nn as nn

from network_blocks import ConvBlock, DeconvBlock, UpBlock, DownBlock

class Net(nn.Module):

    def __init__(self, channels, features, filt):
        super(Net, self).__init__()

        # testing conv blocks
        self.lay0 = ConvBlock(channels, features, 3, 1, 1, activ='prelu')
        self.lay1 = ConvBlock(features, filt, 1, 1, 0, activ='prelu')
        self.lay2 = ConvBlock(filt, filt, 1, 1, 0, activ='prelu')

        kernel = 4
        stride = 2
        padding = 2

        # not final model but was using to test code blocks
        self.up1 = UpBlock(filt, kernel, stride, padding)
        self.down1 = DownBlock(filt, kernel, stride, padding)



    def forward(self, x):
        x = self.lay0(x)
        x = self.lay1(x)
        x = self.lay2(x)


        x = self.cnn_conv(x)
        x = self.cnn_deconv(x)
        return x
