#! /usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Desc: 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from __future__ import division
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch
import torch.utils.model_zoo as model_zoo



class Bottle2neckX(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, baseWidth, cardinality, stride=1, downsample=None, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: base width.
            cardinality: num of convolution groups.
            stride: conv stride. Replaces pooling layer.
            scale: number of scale.
            type: 'normal': normal set. 'stage': frist blokc of a new stage.
        """
        super(Bottle2neckX, self).__init__()

        D = int(math.floor(planes * (baseWidth/64.0)))
        C = cardinality

        # print('inplanes, D*C*scale, D*C, scale, planes', inplanes, D*C*scale,D*C, scale, planes)
        self.conv1 = nn.Conv2d(inplanes, D*C*scale, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.InstanceNorm2d(D*C*scale)

        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride = stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(nn.Conv2d(D*C, D*C, kernel_size=3, stride = stride, padding=1, groups=C, bias=False))
          bns.append(nn.InstanceNorm2d(D*C))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(D*C*scale, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.InstanceNorm2d(planes * 4)
        self.relu = nn.LeakyReLU(inplace=True)

        self.downsample = downsample
        self.width  = D*C
        self.stype = stype
        self.scale = scale

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0 or self.stype=='stage':
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(self.bns[i](sp))
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype=='normal':
          out = torch.cat((out, spx[self.nums]),1)
        elif self.scale != 1 and self.stype=='stage':
          out = torch.cat((out, self.pool(spx[self.nums])),1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Res2NeXt(nn.Module):
    def __init__(self, block, baseWidth, cardinality, layers, num_classes, scale=4):
        """ Constructor
        Args:
            baseWidth: baseWidth for ResNeXt.
            cardinality: number of convolution groups.
            layers: config of layers, e.g., [3, 4, 6, 3]
            num_classes: number of classes
            scale: scale in res2net
        """
        super(Res2NeXt, self).__init__()

        self.cardinality = cardinality
        self.baseWidth = baseWidth
        self.num_classes = num_classes
        self.inplanes = 64
        self.output_size = 64
        self.scale = scale

        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.InstanceNorm2d(64)
        self.relu = nn.LeakyReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], 2)
        self.layer3 = self._make_layer(block, 256, layers[2], 2)
        self.layer4 = self._make_layer(block, 512, layers[3], 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:

            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality, stride, downsample, scale=self.scale, stype='stage'))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# from lib.bmcan_architecture.layers.utils import count_param
# if __name__ == '__main__':
#     images = torch.rand(1, 3, 256, 256)
#     model = Res2NeXt(Bottle2neckX, layers = [3, 4, 6, 3], baseWidth = 4, cardinality=8, scale = 4, num_classes=1000)
#     param = count_param(model)
#     print('net totoal parameters: %.2fM (%d)' % (param / 1e6, param))
#     # model = model.cuda(0)
#     print(model(images).size())

from lib.bmcan_architecture.layers.utils import count_param
from lib.bmcan_architecture.discriminator import SynImageDecoder
from lib.bmcan_architecture.patchnce import PatchSampleMLP, PatchNCELoss

class Classifier_Module(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=False))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out


class BMCANSharedEncoder(nn.Module):
    def __init__(self, in_dim, n_class, layers = [3, 4, 6, 3], baseWidth = 4, cardinality=8, scale = 4):
        super(BMCANSharedEncoder, self).__init__()

        self.in_dim = in_dim
        self.n_class = n_class


        self.cardinality = cardinality
        self.baseWidth = baseWidth

        self.inplanes = 64

        self.scale = scale
        block = Bottle2neckX

        self.conv1 = nn.Conv2d(self.in_dim, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.InstanceNorm2d(64)
        self.relu = nn.LeakyReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], 2)
        self.layer3 = self._make_layer(block, 256, layers[2], 1)
        self.layer4 = self._make_layer(block, 512, layers[3], 1)

        self.out_conv1 = Classifier_Module(inplanes=1024, dilation_series=[2, 6, 12], padding_series=[2, 6, 12], num_classes=self.n_class)
        self.out_conv2 = Classifier_Module(inplanes=2048, dilation_series=[2, 6, 12], padding_series=[2, 6, 12], num_classes=self.n_class)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # print('downsample', self.inplanes, planes * block.expansion,1,stride )

            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(planes * block.expansion),
            )

        layers = []
        # print('planes', planes)
        layers.append(
            block(self.inplanes, planes, self.baseWidth, self.cardinality, stride, downsample, scale=self.scale,
                  stype='stage'))
        self.inplanes = planes * block.expansion
        # print("self.inplanes",  self.inplanes)
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = x

        x = self.maxpool1(x)

        x = self.layer1(x)
        x1 = x

        x = self.layer2(x)
        x2 = x

        x = self.layer3(x)
        x3 = x

        x = self.layer4(x)
        x4 = x

        # print('x0',x0.shape)
        # print('x1',x1.shape)


        out1 = self.out_conv1(x3)
        out2 = self.out_conv2(x4)

        low = x0
        rec = x4
        return low, out1, out2, rec, [x0, x1, x2, x3, x4]


class BMCANOnlySharedEncoder(nn.Module):
    def __init__(self, in_dim, n_class, layers = [3, 4, 6, 3], baseWidth = 4, cardinality=8, scale = 4):
        super(BMCANOnlySharedEncoder, self).__init__()
        """
        num_blocks = [2, 2, 3, 5, 2]            # L
        channels = [64, 96, 192, 384, 768]      # D
        return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=1000)
        """
        self.in_dim = in_dim
        self.n_class = n_class


        self.cardinality = cardinality
        self.baseWidth = baseWidth

        self.inplanes = 64

        self.scale = scale
        block = Bottle2neckX

        self.conv1 = nn.Conv2d(self.in_dim, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.InstanceNorm2d(64)
        self.relu = nn.LeakyReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], 2)
        self.layer3 = self._make_layer(block, 256, layers[2], 1)
        self.layer4 = self._make_layer(block, 512, layers[3], 1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, self.baseWidth, self.cardinality, stride, downsample, scale=self.scale,
                  stype='stage'))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = x

        x = self.maxpool1(x)
        x = self.layer1(x)
        x1 = x

        x = self.layer2(x)
        x2 = x

        x = self.layer3(x)
        x3 = x

        x = self.layer4(x)
        x4 = x

        low = x0
        rec = x4
        return low, x3, x4, rec, [x0, x1, x2, x3, x4]

if __name__ == '__main__':
    n_modal = 1
    n_classes = 5

    shared_encoder = BMCANSharedEncoder(in_dim=n_modal, n_class=n_classes)
    # syn_res_decoder = SynImageDecoder(out_channel=n_modal, shared_code_channel=2048)
    #
    param = count_param(shared_encoder)
    print('net totoal parameters: %.2fM (%d)' % (param / 1e6, param))
    #
    input_tensor = torch.rand(2, n_modal, 256, 256)
    low_s, s_pred1, s_pred2, code_s_common, feat_true_s_list = shared_encoder(input_tensor)
    # syn_res_tensor = syn_res_decoder(code_s_common)
    # print('syn_res_tensor:', syn_res_tensor.shape, syn_res_tensor.min(), syn_res_tensor.max())
