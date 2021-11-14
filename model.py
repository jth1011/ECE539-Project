import torch
import torch.nn as nn

from network_blocks import ConvBlock, DeconvBlock, UpBlock, DownBlock


class Net(nn.Module):

    def __init__(self, channels, filters, features, scale_fact):
        super(Net, self).__init__()

        # testing conv blocks
        self.lay0 = ConvBlock(in_size=channels, out_size=features, kernel=3, stride=1, padding=1,activ='prelu')
        self.lay1 = ConvBlock(in_size=features, out_size=filters, kernel=1, stride=1, padding=0, activ='prelu')
        self.lay2 = ConvBlock(in_size=filters, out_size=filters, kernel=1, stride=1, padding=0, activ='prelu')
        self.out = DeconvBlock(in_size=filters, out_size=features, kernel=1,stride=1,padding=0,activ='prelu')
        self.out2 = DeconvBlock(in_size=features, out_size=channels, kernel=2,stride=2,padding=0,activ='prelu')


    def forward(self, x):
        x = self.lay0(x)
        x = self.lay1(x)
        x = self.lay2(x)
        x = self.out(x)
        x = self.out2(x)

        return x
