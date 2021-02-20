import torch.nn as nn

from base_networks import *


class Net(nn.Module):
    def __init__(self, num_channels, base_filter, feat):
        super(Net, self).__init__()

        kernel = 3
        stride = 0
        padding = 0

        # Initial Feature Extraction
        feat0 = ConvBlock(num_channels, feat, 3, 1, 1, activation='prelu', norm=None)
        feat1 = ConvBlock(feat, base_filter, 1, 1, 0, activation='prelu', norm=None)
        head = [feat0, feat1]

        # Body
        body_part_1 = [
            ResBlock(base_filter, kernel, stride, padding) for _ in range(5)
        ]

        upsample_1 = UpSampleX2(base_filter, 3, 2, 1)
        body_part_2 = [
            ResBlock(base_filter, kernel, stride, padding) for _ in range(4)
        ]

        # Reconstruction
        upsample_2 = UpSampleX2(base_filter, 3, 2, 1)
        output = ConvBlock(base_filter, num_channels, 3, 1, 1, activation=None, norm=None)

        self.head = nn.Sequential(*head)
        self.body_1 = nn.Sequential(*body_part_1)
        self.body_2 = nn.Sequential(upsample_1, *body_part_2)
        self.tail = nn.Sequential(upsample_2, output)

    def forward(self, x):
        x = self.head(x)

        after_body_1 = self.body_1(x)
        after_body_1 += x

        after_body_2 = self.body_2(after_body_1)
        after_body_2 += after_body_1

        out = self.tail(after_body_2)
        return out
