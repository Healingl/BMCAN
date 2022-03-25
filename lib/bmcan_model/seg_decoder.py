#! /usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Desc: 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


import torch
import torch.nn as nn
from torch.nn import init
from lib.bmcan_model.model import *
from lib.bmcan_model.model_util import *

from torch.autograd import Variable

class SegDecoder(nn.Module):
    def __init__(self, output_shape, is_batchnorm=True):
        super(SegDecoder, self).__init__()
        self.output_shape = output_shape
        assert len(self.output_shape) == 4

        self.num_classes = self.output_shape[1]
        self.is_batchnorm = is_batchnorm
        # input tensor
        """
        # [4,1,256,256] -> [4,64,256,256] *nce
        low = self.input_conv(x)
        nce_layer_output_1 = low

        # [4,64,256,256] -> [4,64,65,65] *nce
        x = self.layer0(low)
        nce_layer_output_2 = x

        # [4,64,65,65] -> [4,256,65,65] *nce
        x = self.layer1(x)
        nce_layer_output_3 = x

        # [4,256,65,65] -> [4,512,33,33] *nce
        x = self.layer2(x)
        nce_layer_output_4 = x
        
        # rec: [4,1024,33,33] -> [4,2048,33,33]*nce
        rec = self.layer4(x)
        nce_layer_output_5 = rec
        
        """

        # nce_layer_output_5 [4,2048,33,33]
        self.decoder_conv_layer1 = nn.Sequential(Conv2dBlock(2048, 512, kernel_size=3, stride=1, padding=1, norm='in', activation='lrelu', bias=False),
                                           Conv2dBlock(512, 256, kernel_size=1, stride=1, padding=0, norm='in',
                                                       activation='lrelu', bias=False),

                                           )
        self.decoder_conv_layer2 = nn.Sequential(
            Conv2dBlock(512, 64, kernel_size=3, stride=1, padding=1, norm='in', activation='lrelu', bias=False),
            )

        self.decoder_conv_layer3 = nn.Sequential(
            Conv2dBlock(128, 32, kernel_size=3, stride=1, padding=1, norm='in', activation='lrelu', bias=False),
        )

        self.final = Conv2dBlock(32, self.num_classes, kernel_size=1, stride=1, padding=0, norm='none', activation='none', bias=False)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         m.weight.data.normal_(0, 0.02)
        # self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            # elif isinstance(m, (nn.BatchNorm2d,nn.InstanceNorm2d)):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)

    def forward(self, input_feature_list):
        # [bs,2048,33,33]
        feature_5 = input_feature_list[4]
        # [4,512,33,33]
        feature_4 = input_feature_list[3]
        # [4,256,65,65]
        feature_3 = input_feature_list[2]
        # [4,64,65,65]
        feature_2 = input_feature_list[1]
        # [4,64,256,256]
        feature_1 = input_feature_list[0]

        # [2, 256, 33, 33]
        x = self.decoder_conv_layer1(feature_5)
        upsample_1 = nn.Upsample(size=(feature_3.shape[2], feature_3.shape[3]), mode='bilinear', align_corners=True)
        x = upsample_1(x)

        # [4,512,65,65]
        x = torch.cat((x, feature_3), dim=1)
        # [4,64,65,65]
        x = self.decoder_conv_layer2(x)

        upsample_2 = nn.Upsample(size=(feature_1.shape[2], feature_1.shape[3]), mode='bilinear', align_corners=True)
        x = upsample_2(x)
        x = torch.cat((x, feature_1), dim=1)
        x = self.decoder_conv_layer3(x)

        seg_output = self.final(x)

        return seg_output

def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count
if __name__ == '__main__':
    print('#### Test Case ###')
    batch_size = 2

    input_channels = 1
    n_classes = 5

    img_size = 256

    x = Variable(torch.rand(batch_size, input_channels, img_size, img_size))
    seg_shared = ResNetN(in_dim=input_channels, n_class=n_classes,
                         n=50)
    print(seg_shared)
    param = count_param(seg_shared)
    print('seg_shared totoal parameters: %.2fM (%d)' % (param / 1e6, param))

    _, _, _, _, feat_true_s_list = seg_shared(x)

    for feat in feat_true_s_list:
        print(feat.shape)


    # seg_decoder = SegDecoder(output_shape=(batch_size,n_classes,img_size,img_size))
    # seg_result =seg_decoder(input_feature_list=feat_true_s_list)
    # print('output',seg_result.shape)