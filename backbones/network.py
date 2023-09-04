import math
import torch
import torch.nn as nn
from backbones.module.blocks import SeparableConv2d, Block, UpBlock, DownBlock, UnetDsv3
from backbones.module.scale_atten import scale_atten_convblock


class MyUnet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(MyUnet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        dsv = 240  # 256 细胞  240肺炎和肿瘤  288结肠息肉  320皮肤
        n = 8
        filters = [n*2, n*4, n*8, n*16, n*32, n*64]
        # pool_size = [160, 80, 40, 20]
        pool_size = [120, 60, 30, 15]
        # pool_size = [128, 64, 32, 16]
        # pool_size = [144, 72, 36, 18]
        #
        # start
        self.conv1 = SeparableConv2d(in_channels, filters[0], 3, stride=1, dilation=1)
        self.relu = nn.ReLU(inplace=False)

        # down sample
        self.block1 = Block(filters[0], filters[1])
        self.down1 = DownBlock(kernel_size=3, stride=2, pool_size=pool_size[0])

        self.block2 = Block(filters[1], filters[2])
        self.down2 = DownBlock(kernel_size=3, stride=2, pool_size=pool_size[1])

        self.block3 = Block(filters[2], filters[3])
        self.down3 = DownBlock(kernel_size=3, stride=2, pool_size=pool_size[2])

        self.block4 = Block(filters[3], filters[4])
        self.down4 = DownBlock(kernel_size=3, stride=2, pool_size=pool_size[3])

        self.block_center = Block(filters[4], filters[5])
        # self.block_center = ULSAM(filters[4], filters[5], 15, 15, 4)

        # up sample
        self.up4 = UpBlock(filters[5], filters[4])
        self.up4_conv = Block(filters[5], filters[4])

        self.up3 = UpBlock(filters[4], filters[3])
        self.up3_conv = Block(filters[4], filters[3])

        self.up2 = UpBlock(filters[3], filters[2])
        self.up2_conv = Block(filters[3], filters[2])

        self.up1 = UpBlock(filters[2], filters[1])
        self.up1_conv = Block(filters[2], filters[1])

        # deep supervision
        self.dsv4 = UnetDsv3(filters[4], 4, dsv)
        self.dsv3 = UnetDsv3(filters[3], 4, dsv)
        self.dsv2 = UnetDsv3(filters[2], 4, dsv)
        self.dsv1 = nn.Conv2d(filters[1], 4, 1)
        # self.scale_att = scale_atten_convblock(in_size=16, out_size=4)
        # # final conv (without any concat)
        # self.final = nn.Sequential(
        #     nn.Conv2d(4, out_channels, 1),
        #     nn.Sigmoid()
        # )

        self.out = nn.Sequential(
            SeparableConv2d(filters[1], out_channels, 1, stride=1, dilation=1),
            nn.Sigmoid()
        )

        # Init weights
        self._init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        e1 = self.block1(x)
        down1 = self.down1(e1)

        e2 = self.block2(down1)
        down2 = self.down2(e2)

        e3 = self.block3(down2)
        down3 = self.down3(e3)

        e4 = self.block4(down3)
        down4 = self.down4(e4)

        center = self.block_center(down4)

        up4 = self.up4(center)
        d4 = torch.cat((e4, up4), dim=1)
        d4 = self.up4_conv(d4)

        up3 = self.up3(d4)
        d3 = torch.cat((e3, up3), dim=1)
        d3 = self.up3_conv(d3)

        up2 = self.up2(d3)
        d2 = torch.cat((e2, up2), dim=1)
        d2 = self.up2_conv(d2)

        up1 = self.up1(d2)
        d1 = torch.cat((e1, up1), dim=1)
        d1 = self.up1_conv(d1)

        # Deep Supervision
        # dsv4 = self.dsv4(d4)
        # dsv3 = self.dsv3(d3)
        # dsv2 = self.dsv2(d2)
        # dsv1 = self.dsv1(d1)
        # dsv_cat = torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1)
        # out = self.scale_att(dsv_cat)
        # final = self.final(out)

        final = self.out(d1)

        return final

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == "__main__":
    from thop import profile
    model = MyUnet(3, 1)
    input = torch.rand(2, 3, 240, 240)
    output = model(input)
    flops, params = profile(model, inputs=(input,))
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
    print("\n")
    # print(model)
