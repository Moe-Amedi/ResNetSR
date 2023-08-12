import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return out

class ResNetSR(nn.Module):
    def __init__(self, num_blocks=16, upscale_factor=4, in_channels=3, out_channels=3):
        super(ResNetSR, self).__init__()

        self.conv_input = nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4, bias=True)
        self.prelu = nn.PReLU()

        self.res_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.res_blocks.append(Block(64, 64))

        self.conv_mid = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn_mid = nn.BatchNorm2d(256)

        self.upscale1 = nn.Sequential(
                    nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=True),
                    nn.BatchNorm2d(256),
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.PReLU(),
                )
        
        self.upscale2 = nn.Sequential(
                    nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=True),
                    nn.BatchNorm2d(256),
                    nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.PReLU(),
                )

        self.conv_output = nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4, bias=True)

    def forward(self, x):
        out = self.conv_input(x)
        out = self.prelu(out)
        residual = out

        for block in self.res_blocks:
            out = block(out)
            out += residual
            residual = out

        out = self.conv_mid(out)
        out = self.bn_mid(out)

        out = self.upscale1(out)

        out = self.upscale2(out)

        out = self.conv_output(out)

        return out

def test():
    net = ResNetSR().cpu()
    x = torch.randn(1, 3, 256, 256).cpu()
    y = net(x).to('cpu')
    print(y.shape)
    
#test()