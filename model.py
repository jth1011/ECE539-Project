import torch
import torch.nn as nn

from network_blocks import ConvBlock, DeconvBlock, UpBlock, DownBlock


class Net(nn.Module):

    def __init__(self, channels, filters, features, scale_fact):
        super(Net, self).__init__()

        if scale_fact == 2:
            kernel = 6
            stride = 2
            padding = 2
        elif scale_fact == 4:
            kernel = 8
            stride = 4
            padding = 2

        # adding initial convolution block to extract features from original image
        self.lay0 = ConvBlock(in_size=channels, out_size=features, kernel=3, stride=1, padding=1,activ='prelu')
        self.lay1 = ConvBlock(in_size=features, out_size=filters, kernel=1, stride=2, padding=0, activ='prelu')
        self.lay2 = ConvBlock(in_size=filters, out_size=filters, kernel=kernel, stride=stride, padding=padding, activ='prelu')
        self.lay3 = DeconvBlock(in_size=filters, out_size=filters, kernel=kernel, stride=stride, padding=padding, activ='prelu')
        self.lay4 = DeconvBlock(in_size=filters, out_size=features, kernel=4,stride=2,padding=0,activ='prelu')
        self.lay5 = DeconvBlock(in_size=features, out_size=16*channels, kernel=3,stride=1,padding=1,activ='prelu')
        self.lay6 = DeconvBlock(in_size=16*channels, out_size=channels, kernel=2, stride=2, padding=0, activ='prelu')


    def forward(self, x):
        x = self.lay0(x)
        x = self.lay1(x)
        x = self.lay2(x)
        x = self.lay3(x)
        x = self.lay4(x)
        x = self.lay5(x)
        x = self.lay6(x)

        return x
