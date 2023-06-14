from math import sqrt
import torch
import torch.nn.functional as F
from torch import nn


class HrNet(nn.Module):


    def __init__(self, input_size:int=512):
        super(HrNet, self).__init__()

        self.stage_1_conv_1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )
        self.stage_1_conv_2 = nn.Conv2d(
            in_channels=8,
            out_channels=24,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )
        self.stage_1_conv_3 = nn.Conv2d(
            in_channels=24,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )
        self.stage_1_conv_4 = nn.Conv2d(
            in_channels=8,
            out_channels=24,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )  
        self.stage_1_conv_5 = nn.Conv2d(
            in_channels=24,
            out_channels=24,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )
        self.stage_1_out = nn.Sequential(
            nn.Conv2d(
                in_channels=24,
                out_channels=8,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            ),
            nn.Softmax(dim=1),
            nn.Conv2d(
                in_channels=8,
                out_channels=8,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            ),
        )

        # stage_2
        self.stage_2_conv_1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )
        self.stage_2_conv_2 = nn.Conv2d(
            in_channels=16,
            out_channels=48,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )
        self.stage_2_conv_3 = nn.Conv2d(
            in_channels=48,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )
        self.stage_2_conv_4 = nn.Conv2d(
            in_channels=16,
            out_channels=40,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )
        self.stage_2_conv_5 = nn.Conv2d(
            in_channels=40,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )
        self.stage_2_up_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=48,
                out_channels=24,
                kernel_size=1,
            ),
            nn.Upsample(
                size=(input_size,input_size)
            )
        )
        self.stage_2_up_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=8,
                kernel_size=1,
            ),
            nn.Upsample(
                size=(input_size,input_size)
            )
        )


    def forward(self, inputs):
        img_size = int(inputs.shape[-1]/2)

        half_input = nn.functional.interpolate(inputs, (img_size, img_size), mode='bilinear')

        s1_1 = self.stage_1_conv_1(inputs)
        s1_2 = self.stage_1_conv_2(s1_1)
        s2_1 = self.stage_2_conv_1(half_input)
        s2_2 = self.stage_2_conv_2(s2_1)
        s2_up_1 = self.stage_2_up_1(s2_2) 

        s1_3_input = s1_2 + s2_up_1

        s1_3 = self.stage_1_conv_3(s1_3_input)
        s2_3 = self.stage_2_conv_3(s2_2)
        s2_4 = self.stage_2_conv_4(s2_3)
        s2_5 = self.stage_2_conv_5(s2_4)
        s2_up_2 = self.stage_2_up_2(s2_5)

        s1_4_input = s1_3 + s2_up_2

        s1_4 = self.stage_1_conv_4(s1_4_input)
        s1_5 = self.stage_1_conv_5(s1_4)
        s1_out = self.stage_1_out(s1_5)

        return s1_out, s1_5


if __name__ == '__main__':

    from torchsummary import summary

    model = HrNet()
    inputs = torch.randn((1,1,1024,1024))

    x = model(inputs)
    print(x.shape)