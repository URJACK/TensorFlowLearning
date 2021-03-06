from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

WORKSPACEDIR = '.\\model'
DATADIR = 'D:\\Storage\\datasets\\td_ai_spring2020\\hw3_data\\faces'


# 0 ~ 1
def pearson(a, b):
    top = torch.mean((a - torch.mean(a)) * (b - torch.mean(b)))
    bottom = torch.std(a) * torch.std(b)
    return top / bottom


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    """
        input (N, in_dim)
        output (N, 3, 64, 64)
        """

    def __init__(self, in_dim, dim=64):
        super(Generator, self).__init__()

        def dconv_bn_relu(in_dim_val, out_dim_val):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim_val, out_dim_val, 5, 2,
                                   padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim_val),
                nn.ReLU())

        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU())
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),
            nn.Tanh())
        self.apply(weights_init)

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y


class Encoder(nn.Module):
    """
    input (N, 3, 64, 64)
    output (N, )
    """

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, noise_dim, dim=64):
        super(Encoder, self).__init__()

        self.noise_dim = noise_dim

        def conv_bn_lrelu(in_dim_val, out_dim_val):
            return nn.Sequential(
                nn.Conv2d(in_dim_val, out_dim_val, 5, 2, 2),
                nn.BatchNorm2d(out_dim_val),
                nn.LeakyReLU(0.2))

        self.ls = nn.Sequential(
            nn.Conv2d(3, dim, 5, 2, 2), nn.LeakyReLU(0.2),
            conv_bn_lrelu(dim, dim * 2),
            conv_bn_lrelu(dim * 2, dim * 4),
            conv_bn_lrelu(dim * 4, dim * 8),
            nn.Conv2d(dim * 8, noise_dim, 4),
            nn.ReLU())
        self.apply(weights_init)

    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1, self.noise_dim)
        return y


class Terminator(nn.Module):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, noise_dim):
        super(Terminator, self).__init__()

        self.work = nn.Sequential(
            nn.Linear(noise_dim, 1),
            nn.Sigmoid()
        )
        self.apply(weights_init)

    def forward(self, x):
        y = self.work(x)
        y = y.view(-1)
        return y
