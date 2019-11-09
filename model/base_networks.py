"""
All base network model included such like generator_A2B, Discriminator_A.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, input_nc):
        super(ResidualBlock, self).__init__()

        sequence = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(input_nc, input_nc, 3),
                        nn.InstanceNorm2d(input_nc),
                        nn.ReLU(True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(input_nc,input_nc, 3),
                        nn.InstanceNorm2d(input_nc)  ]

        self.conv_block = nn.Sequential(*sequence)

    def forward(self, x):
        return x + self.conv_block(x)

class G(nn.Module):
    def __init__(self, input_nc, output_nc, n_blocks=6):
        super(G, self).__init__()

        # Initial convolution block       
        model = [  nn.ReflectionPad2d(3),
                   nn.Conv2d(input_nc, 64, 7),
                   nn.InstanceNorm2d(64),
                   nn.ReLU(inplace=True) ]

        # Downsampling
        ngf = 64
        n_downSampling = 2
        for i in range(n_downSampling):
            mult = 2**i
            model += [ nn.Conv2d(ngf*mult, ngf*mult*2, 3, stride=2, padding=1),
                       nn.InstanceNorm2d(ngf*mult),
                       nn.ReLU(True) ]

        # Residual blocks
        for _ in range(n_blocks):
            model += [ResidualBlock(ngf*(2**n_downSampling))]

        # Upsampling
        for i in range(2):
            mult = 2**(n_downSampling-i)
            model += [ nn.ConvTranspose2d(ngf*mult,int(ngf*mult//2), 3, stride=2, padding=1, output_padding=1),
                       nn.InstanceNorm2d(int(ngf*mult//2)),
                       nn.ReLU(True) ]

        # Output layer
        model += [ nn.ReflectionPad2d(3) ]
        model += [ nn.Conv2d(64, output_nc, 7) ]
        model += [ nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class D(nn.Module):
    def __init__(self, input_nc):
        super(D, self).__init__()

        model = [ nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                  nn.LeakyReLU(0.2, inplace=True) ]

        ngf = 64
        for i in range(3):
            mult = 2**i
            model += [ nn.Conv2d(ngf*mult, ngf*mult*2, 4, stride=2, padding=1),
                       nn.InstanceNorm2d(ngf*mult*2), 
                       nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [ nn.Conv2d(512, 1, 4, padding=1) ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


