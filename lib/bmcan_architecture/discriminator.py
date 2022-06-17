#! /usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Desc: 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import functools
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.models as models


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride=1,
                 padding=0, dilation=1, norm='none', activation='relu', pad_type='zero', bias=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = bias
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        # else:
        # assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, dilation=dilation, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class SynImageDecoder(nn.Module):
    def __init__(self,  out_channel, shared_code_channel):
        super(SynImageDecoder, self).__init__()



        self.shared_code_channel = shared_code_channel

        self.feature_chns = [32, 64, 128, 256]

        self.main = []

        self.main += [Conv2dBlock(self.shared_code_channel, self.feature_chns[3], 3, stride=1, padding=1, norm='in', activation='lrelu',
                                  pad_type='reflect', bias=False)]
        self.main += [ResBlocks(3, self.feature_chns[3], 'bn', 'relu', pad_type='zero')]

        self.upsample = nn.Sequential(

            # input: 1/8 * 1/8
            nn.ConvTranspose2d(self.feature_chns[3], self.feature_chns[3], 4, 2, 1, bias=False),
            nn.InstanceNorm2d(self.feature_chns[3]),
            nn.LeakyReLU(True),
            Conv2dBlock(self.feature_chns[3], self.feature_chns[2], 3, 1, 1, norm='in', activation='lrelu', pad_type='zero'),

            # 1/4 * 1/4
            nn.ConvTranspose2d(self.feature_chns[2], self.feature_chns[2], 4, 2, 1, bias=False),
            nn.InstanceNorm2d(self.feature_chns[2]),
            nn.LeakyReLU(True),
            Conv2dBlock(self.feature_chns[2], self.feature_chns[1], 3, 1, 1, norm='in', activation='lrelu', pad_type='zero'),

            # 1/2 * 1/2
            nn.ConvTranspose2d(self.feature_chns[1], self.feature_chns[1], 4, 2, 1, bias=False),
            nn.InstanceNorm2d(self.feature_chns[1]),
            nn.LeakyReLU(True),
            Conv2dBlock(self.feature_chns[1], self.feature_chns[0], 3, 1, 1, norm='in', activation='lrelu',
                        pad_type='zero'),
            # 1 * 1
            nn.Conv2d(self.feature_chns[0], out_channel, 3, 1, 1)
        )
        self.main += [self.upsample]

        self.main = nn.Sequential(*self.main)

    def forward(self, shared_code):
        output = self.main(shared_code)
        output = torch.tanh(output)
        return output


class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PatchGANDiscriminator, self).__init__()
        # PatchGAN

        self.dis_layer_1 = Conv2dBlock(in_dim, 64, kernel_size=4, stride=2, padding=1, norm='none', activation='lrelu',
                                       bias=False)
        self.dis_layer_2 = Conv2dBlock(64, 128, kernel_size=4, stride=2, padding=1, norm='in', activation='lrelu',
                                       bias=False)
        self.dis_layer_3 = Conv2dBlock(128, 256, kernel_size=4, stride=2, padding=1, norm='in', activation='lrelu',
                                       bias=False)
        self.dis_layer_4 = Conv2dBlock(256, 512, kernel_size=4, stride=2, padding=1, norm='in', activation='lrelu',
                                       bias=False)
        self.dis_layer_5 = nn.Conv2d(512, out_dim, 1, padding=0)


    def forward(self, x):
        x = self.dis_layer_1(x)
        # print(x.size())
        x = self.dis_layer_2(x)
        # print(x.size())
        x = self.dis_layer_3(x)
        # print(x.size())
        x = self.dis_layer_4(x)
        # print(x.size())
        x = self.dis_layer_5(x)
        # print(x.size())
        return x
