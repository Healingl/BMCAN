import functools
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.models as models

from torch.autograd import Variable
from lib.bmcan_model.model_util import *
from lib.bmcan_model.seg_model  import DeeplabMulti, ResNet

pspnet_specs = {
    'n_classes': 5,
    'input_size': (713, 713),
    'block_config': [3, 4, 23, 3],
}
'''
Sequential blocks
'''
class ResNetNOrigin(nn.Module):         #使用的是resnet50   而不是resnet101   其他结构一样
    def __init__(self, in_dim, n_class, n=50):
        super(ResNetNOrigin, self).__init__()
        layers = {
            '35': [3, 4, 2, 2],
            "50": [3, 4, 6, 3],
            "101": [3, 4, 23, 3],
            "152": [3, 8, 36, 3]
        }
        self.in_dim = in_dim
        self.n_classes = n_class
        print(layers[str(n)])
        Seg_Model = ResNet(in_dim, self.n_classes, layers[str(n)])

        self.layer0 = nn.Sequential(Seg_Model.conv1, Seg_Model.bn1, Seg_Model.relu, Seg_Model.maxpool)
        self.layer1 = Seg_Model.layer1
        self.layer2 = Seg_Model.layer2
        self.layer3 = Seg_Model.layer3
        self.layer4 = Seg_Model.layer4

        self.final1 = Seg_Model.layer5
        self.final2 = Seg_Model.layer6

    def forward(self, x):
        inp_shape = x.shape[2:]
        # [4,1,256,256] -> [4,64,65,65] *
        low = self.layer0(x)
        nce_layer_output_1 = low

        # [4,64,65,65] -> [4,256,65,65] *
        x = self.layer1(low)

        nce_layer_output_2 = x
        # [4,256,65,65] -> [4,512,33,33] *
        x = self.layer2(x)
        nce_layer_output_3 = x
        # [4,512,65,65] -> [4,1024,33,33]*
        x = self.layer3(x)
        nce_layer_output_4 = x

        # x1: [4,5,33,33]
        x1 = self.final1(x)

        # rec: [4,1024,33,33] -> [4,2048,33,33]*
        rec = self.layer4(x)
        nce_layer_output_5 = rec

        # x2: [4,5,33,33]
        x2 = self.final2(rec)

        return low, x1, x2, rec, [nce_layer_output_1, nce_layer_output_2, nce_layer_output_3,nce_layer_output_4,nce_layer_output_5]

    def get_1x_lr_params_NOscale(self):
        b = []

        b.append(self.layer0)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        b = []
        b.append(self.final1.parameters())
        b.append(self.final2.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, learning_rate):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr':  1 * learning_rate},
                {'params': self.get_10x_lr_params(),        'lr': 10 * learning_rate}]


class ResNetN(nn.Module):         #使用的是resnet50   而不是resnet101   其他结构一样
    def __init__(self, in_dim, n_class, n=50):
        super(ResNetN, self).__init__()
        layers = {
            '35': [3, 4, 2, 2],
            # 3,4,6,3 is resblock num
            "50": [3, 4, 6, 3],
            "101": [3, 4, 23, 3],
            "152": [3, 8, 36, 3]
        }
        self.in_dim = in_dim
        self.n_classes = n_class
        print(layers[str(n)])

        first_layer_filters = 64
        self.input_conv = Conv2dBlock(in_dim, first_layer_filters, 3, 1, 1, norm='in', activation='lrelu',bias=False)

        Seg_Model = ResNet(first_layer_filters, self.n_classes, layers[str(n)])

        self.layer0 = nn.Sequential(Seg_Model.conv1, Seg_Model.bn1, Seg_Model.relu, Seg_Model.maxpool)
        self.layer1 = Seg_Model.layer1
        self.layer2 = Seg_Model.layer2
        self.layer3 = Seg_Model.layer3
        self.layer4 = Seg_Model.layer4

        self.final1 = Seg_Model.layer5
        self.final2 = Seg_Model.layer6

        # self._init_weights()
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         m.weight.data.normal_(0, 0.02)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            # elif isinstance(m, (nn.BatchNorm2d,nn.InstanceNorm2d)):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)

    def forward(self, x):
        inp_shape = x.shape[2:]

        # [4,1,256,256] -> [4,64,256,256] *nce
        low = self.input_conv(x)
        nce_layer_output_1 = low
        print('layer1',low.shape)
        # [4,64,256,256] -> [4,64,65,65] *nce
        x = self.layer0(low)
        nce_layer_output_2 = x
        print('layer2', x.shape)
        # [4,64,65,65] -> [4,256,65,65] *nce
        x = self.layer1(x)
        nce_layer_output_3 = x
        print('layer3', x.shape)
        # [4,256,65,65] -> [4,512,33,33] *nce
        x = self.layer2(x)
        nce_layer_output_4 = x

        # [4,512,65,65] -> [4,1024,33,33]
        x = self.layer3(x)


        # x1: [4,5,33,33]
        x1 = self.final1(x)

        # rec: [4,1024,33,33] -> [4,2048,33,33]*nce
        rec = self.layer4(x)
        nce_layer_output_5 = rec

        # x2: [4,5,33,33]
        x2 = self.final2(rec)

        return low, x1, x2, rec, [nce_layer_output_1, nce_layer_output_2, nce_layer_output_3, nce_layer_output_4, nce_layer_output_5]

    def get_1x_lr_params_NOscale(self):
        b = []

        b.append(self.layer0)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        b = []
        b.append(self.final1.parameters())
        b.append(self.final2.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, learning_rate):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr':  1 * learning_rate},
                {'params': self.get_10x_lr_params(),        'lr': 10 * learning_rate}]


class SharedResNet100(nn.Module):
    def __init__(self, in_dim, n_class=pspnet_specs['n_classes'], pretrained=False):
        super(SharedResNet100, self).__init__()
        self.in_dim = in_dim
        self.n_classes = n_class

        Seg_Model = DeeplabMulti(pretrained=pretrained, in_dim=in_dim, num_classes=self.n_classes)

        self.layer0 = nn.Sequential(Seg_Model.conv1, Seg_Model.bn1, Seg_Model.relu, Seg_Model.maxpool)
        self.layer1 = Seg_Model.layer1
        self.layer2 = Seg_Model.layer2
        self.layer3 = Seg_Model.layer3
        self.layer4 = Seg_Model.layer4

        self.final1 = Seg_Model.layer5
        self.final2 = Seg_Model.layer6

    def forward(self, x):
        inp_shape = x.shape[2:]

        low = self.layer0(x)
        # [2, 64, 65, 129]
        x = self.layer1(low)
        x = self.layer2(x)

        x = self.layer3(x)
        x1 = self.final1(x)

        rec = self.layer4(x)
        x2 = self.final2(rec)

        return low, x1, x2, rec

    def get_1x_lr_params_NOscale(self):
        b = []

        b.append(self.layer0)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        b = []
        b.append(self.final1.parameters())
        b.append(self.final2.parameters())

        for j in range(len(b)):
            b.append(self.final2.parameters())
            for i in b[j]:
                yield i

    def optim_parameters(self, learning_rate):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr':  1 * learning_rate},
                {'params': self.get_10x_lr_params(),        'lr': 10 * learning_rate}]

class Classifier(nn.Module):
    def __init__(self, inp_shape):
        super(Classifier, self).__init__()
        n_classes = pspnet_specs['n_classes']
        self.inp_shape = inp_shape

        # PSPNet_Model = PSPNet(pretrained=True)

        self.dropout = nn.Dropout2d(0.1)
        self.cls     = nn.Conv2d(512, n_classes, kernel_size=1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.cls(x)
        x = F.upsample(x, size=self.inp_shape, mode='bilinear')
        return x


class SharedEncoder(nn.Module):
    def __init__(self, input_channels, code_size):
        super(SharedEncoder, self).__init__()
        self.input_channels = input_channels
        self.code_size = code_size

        self.cnn = nn.Sequential(nn.Conv2d(self.input_channels, 64, 7, stride=2, padding=3), # 128 * 256
                                nn.InstanceNorm2d(64),
                                nn.ReLU(),
                                nn.Conv2d(64, 128, 3, stride=2, padding=1), # 64 * 128
                                nn.InstanceNorm2d(128),
                                nn.ReLU(),
                                nn.Conv2d(128, 256, 3, stride=2, padding=1), # 32 * 64
                                nn.InstanceNorm2d(256),
                                nn.ReLU(),
                                nn.Conv2d(256, 256, 3, stride=2, padding=1), # 16 * 32
                                nn.InstanceNorm2d(256),
                                nn.ReLU(),
                                nn.Conv2d(256, 256, 3, stride=2, padding=1), # 8 * 16
                                nn.InstanceNorm2d(256),
                                nn.ReLU())
        self.model = []
        self.model += [self.cnn]
        self.model += [nn.AdaptiveAvgPool2d((1, 1))]
        self.model += [nn.Conv2d(256, code_size, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, 0.02)
        #self.pooling = nn.AvgPool2d(4)

        #self.fc = nn.Sequential(nn.Conv2d(128, code_size, 1, 1, 0))

    def forward(self, x):
        bs = x.size(0)
        #feats = self.model(x)
        #feats = self.pooling(feats)

        output = self.model(x).view(bs, -1)

        return output


class PrivateEncoder(nn.Module):
    def __init__(self, input_channels, code_size):
        super(PrivateEncoder, self).__init__()
        self.input_channels = input_channels
        self.code_size = code_size

        self.cnn = nn.Sequential(nn.Conv2d(self.input_channels, 64, 7, stride=2, padding=3), # 128 * 256
                                nn.BatchNorm2d(64),
                                nn.ReLU(),
                                nn.Conv2d(64, 128, 3, stride=2, padding=1), # 64 * 128
                                nn.BatchNorm2d(128),
                                nn.ReLU(),
                                nn.Conv2d(128, 256, 3, stride=2, padding=1), # 32 * 64
                                nn.BatchNorm2d(256),
                                nn.ReLU(),
                                nn.Conv2d(256, 256, 3, stride=2, padding=1), # 16 * 32
                                nn.BatchNorm2d(256),
                                nn.ReLU(),
                                nn.Conv2d(256, 256, 3, stride=2, padding=1), # 8 * 16
                                nn.BatchNorm2d(256),
                                nn.ReLU())
        self.model = []
        self.model += [self.cnn]
        self.model += [nn.AdaptiveAvgPool2d((1, 1))]
        self.model += [nn.Conv2d(256, code_size, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, 0.02)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #         # for i in m.parameters():
        #         #     i.requires_grad = False
        #self.pooling = nn.AvgPool2d(4)

        #self.fc = nn.Sequential(nn.Conv2d(128, code_size, 1, 1, 0))

    def forward(self, x):
        bs = x.size(0)
        #feats = self.model(x)
        #feats = self.pooling(feats)

        output = self.model(x).view(bs, -1)

        return output


class SharedDecoder(nn.Module):
    def __init__(self, shared_code_channel, private_code_size, out_channel):
        super(SharedDecoder, self).__init__()
        num_att = 256
        self.shared_code_channel = shared_code_channel
        self.private_code_size = private_code_size

        self.main = []
        self.upsample = nn.Sequential(
            # input: 1/8 * 1/8
            nn.ConvTranspose2d(256, 256, 4, 2, 2, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
            Conv2dBlock(256, 128, 3, 1, 1, norm='ln', activation='relu', pad_type='zero'),
            # 1/4 * 1/4
            nn.ConvTranspose2d(128, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            Conv2dBlock(128, 64 , 3, 1, 1, norm='ln', activation='relu', pad_type='zero'),
            # 1/2 * 1/2
            nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            Conv2dBlock(64 , 32 , 3, 1, 1, norm='ln', activation='relu', pad_type='zero'),
            # 1 * 1
            nn.Conv2d(32, out_channel, 3, 1, 1),
            nn.Tanh())

        self.main += [Conv2dBlock(shared_code_channel+num_att+1, 256, 3, stride=1, padding=1, norm='ln', activation='relu', pad_type='reflect', bias=False)]
        self.main += [ResBlocks(3, 256, 'ln', 'relu', pad_type='zero')]
        self.main += [self.upsample]

        self.main = nn.Sequential(*self.main)
        self.mlp_att   = nn.Sequential(nn.Linear(private_code_size*2, private_code_size),
                             nn.ReLU(),
                             nn.Linear(private_code_size, private_code_size),
                             nn.ReLU(),
                             nn.Linear(private_code_size, private_code_size),
                             nn.ReLU(),
                             nn.Linear(private_code_size, num_att))
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, 0.02)

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = torch.exp(adain_params[:, m.num_features:2*m.num_features])
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params

    def forward(self, shared_code, private_code, d):
        d = torch.FloatTensor(shared_code.shape[0], 1).fill_(d)
        if torch.cuda.is_available():
            d = d.cuda()
        d = d.unsqueeze(1)
        d_img = d.view(d.size(0), d.size(1), 1, 1).expand(d.size(0), d.size(1), shared_code.size(2), shared_code.size(3))
        noise = torch.randn_like(private_code)#;print(torch.cat([private_code, noise], 1).shape)
        # print(private_code.shape)
        att_params = self.mlp_att(torch.cat([private_code, noise], 1))
        att_img    = att_params.view(att_params.size(0), att_params.size(1), 1, 1).expand(att_params.size(0), att_params.size(1), shared_code.size(2), shared_code.size(3))
        code         = torch.cat([shared_code, att_img, d_img], 1)

        output = self.main(code)
        return output


class PrivateDecoder(nn.Module):
    def __init__(self, shared_code_channel, out_channel, skip=False):
        super(PrivateDecoder, self).__init__()
        num_att = 256
        self.shared_code_channel = shared_code_channel
        self.skip =skip
        self.main = []
        self.upsample = nn.Sequential(
            # input: 1/8 * 1/8
            nn.ConvTranspose2d(256, 256, 4, 2, 2, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
            Conv2dBlock(256, 128, 3, 1, 1, norm='in', activation='lrelu', pad_type='zero'),

            # 1/4 * 1/4
            nn.ConvTranspose2d(128, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            Conv2dBlock(128, 64, 3, 1, 1, norm='in', activation='lrelu', pad_type='zero'),

            # 1/2 * 1/2
            nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            Conv2dBlock(64, 32, 3, 1, 1, norm='in', activation='lrelu', pad_type='zero'),

            # 1 * 1
            nn.Conv2d(32, out_channel, 3, 1, 1)
        )

        self.main += [Conv2dBlock(shared_code_channel, 256, 3, stride=1, padding=1, norm='in', activation='lrelu', pad_type='reflect', bias=False)]
        self.main += [ResBlocks(3, 256, 'in', 'lrelu', pad_type='zero')]
        self.main += [self.upsample]

        self.main = nn.Sequential(*self.main)
        # self._init_weights()
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, 0.02)
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            # elif isinstance(m, (nn.BatchNorm2d,nn.InstanceNorm2d)):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)

    def forward(self, shared_code, inputimg):
        output = self.main(shared_code)
        if self.skip:
            output += inputimg
        output = torch.tanh(output)
        return output


class DomainClassifier(nn.Module):
    def __init__(self, in_dim, out_dim=1):
        super(DomainClassifier, self).__init__()
        # FCN classification layer

        self.feature = nn.Sequential(
            Conv2dBlock(in_dim, 64, 4, stride=2, padding=1, norm='none', activation='lrelu', bias=False),
            Conv2dBlock(64, 128, 4, stride=2, padding=1, norm='in', activation='lrelu', bias=False),
            Conv2dBlock(128, 256, 4, stride=2, padding=1, norm='in', activation='lrelu', bias=False),
            Conv2dBlock(256, 512, 4, stride=2, padding=1, norm='in', activation='lrelu', bias=False),
            nn.Conv2d(512, out_dim, 4, stride=1, padding=2)
        )
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, 0.02)

    def forward(self, x):
        x = self.feature(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Discriminator, self).__init__()
        # FCN classification layer

        self.feature = nn.Sequential(
            Conv2dBlock(in_dim, 64, 4, stride=2, padding=1, norm='none', activation='lrelu', bias=False),
            Conv2dBlock(64, 128, 4, stride=2, padding=1, norm='in', activation='lrelu', bias=False),
            Conv2dBlock(128, 256, 4, stride=2, padding=1, norm='in', activation='lrelu', bias=False),
            Conv2dBlock(256, 512, 4, stride=2, padding=1, norm='in', activation='lrelu', bias=False),
            nn.Conv2d(512, out_dim, 1, padding=0),
            # nn.Sigmoid()
        ) 
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, 0.02)

    def forward(self, x):   
        x = self.feature(x)
        return x

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PatchGANDiscriminator, self).__init__()
        # PatchGAN

        self.dis_layer_1 = Conv2dBlock(in_dim, 64, kernel_size=4, stride=2, padding=1, norm='none', activation='lrelu', bias=False)
        self.dis_layer_2 = Conv2dBlock(64, 128, kernel_size=4, stride=2, padding=1, norm='in', activation='lrelu', bias=False)
        self.dis_layer_3 = Conv2dBlock(128, 256, kernel_size=4, stride=2, padding=1, norm='in', activation='lrelu', bias=False)
        self.dis_layer_4 = Conv2dBlock(256, 512, kernel_size=4, stride=2, padding=1, norm='in', activation='lrelu', bias=False)
        self.dis_layer_5 = nn.Conv2d(512, out_dim, 1, padding=0)
        # self._init_weights()
        # self.feature = nn.Sequential(
        #     # (64,4,2,1)
        #     Conv2dBlock(in_dim, 64, kernel_size=4, stride=2, padding=1, norm='none', activation='lrelu', bias=False),
        #     Conv2dBlock(64, 128, kernel_size=4, stride=2, padding=1, norm='in', activation='lrelu', bias=False),
        #     Conv2dBlock(128, 256, kernel_size=4, stride=2, padding=1, norm='in', activation='lrelu', bias=False),
        #     Conv2dBlock(256, 512, kernel_size=4, stride=2, padding=1, norm='in', activation='lrelu', bias=False),
        #     Conv2dBlock(512, out_dim, kernel_size=4, stride=1, padding=1,norm='in', activation='lrelu', bias=False),
        #     nn.Conv2d(out_dim, out_dim, kernel_size=4, stride=1, padding=1,bias=False),
        #     # nn.Sigmoid()
        # )
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, 0.02)
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            # elif isinstance(m, (nn.BatchNorm2d,nn.InstanceNorm2d)):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)

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



# Origin
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        # k,s,p: 4, 2, 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        # (k,s,p: 4, 2, 1)*2
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


class InfoQ(nn.Module):
    def __init__(self, in_dim, common_code_size):
        super(InfoQ, self).__init__()
        # FCN classification layer
        self.feature = nn.Sequential(
            Conv2dBlock(in_dim, 64, 6, stride=2, padding=2, norm='none', activation='lrelu', bias=False),
            Conv2dBlock(64, 128, 4, stride=2, padding=1, norm='in', activation='lrelu', bias=False),

        )
        self.domain_classifier = nn.Sequential(
            Conv2dBlock(128, 256, 4, stride=2, padding=1, norm='in', activation='lrelu', bias=False),
            Conv2dBlock(256, 512, 4, stride=2, padding=1, norm='in', activation='lrelu', bias=False),
            nn.Conv2d(512, 1, 1, padding=0),  # nn.Sigmoid()
        )
        self.semantic_code = nn.Sequential(
            Conv2dBlock(128, 256, 2, stride=2, padding=1, norm='in', activation='lrelu', bias=False),
            Conv2dBlock(256, 128, 3, stride=1, padding=1, norm='in', activation='lrelu', bias=False),
            nn.Conv2d(128, 2 * common_code_size, 1, padding=0)
        )

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, 0.02)

    def forward(self, x):
        x = self.feature(x)
        semantic_code = self.semantic_code(x)
        domain = self.domain_classifier(x).squeeze()
        # x = self.global_pooling(x).view(-1)
        return domain, semantic_code


class CodeClassifier(nn.Module):
    def __init__(self, in_dim, out_dim=1):
        super(CodeClassifier, self).__init__()
        # FCN classification layer

        self.feature = nn.Sequential(
            Conv2dBlock(in_dim, 512, 4, stride=2, padding=1, norm='none', activation='lrelu', bias=False),
            Conv2dBlock(512, 256, 4, stride=2, padding=1, norm='none', activation='lrelu', bias=False),
            Conv2dBlock(256, 64, 4, stride=2, padding=1, norm='none', activation='lrelu', bias=False),
            nn.Conv2d(64, out_dim, 4, padding=2))
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, 0.02)

    def forward(self, x):
        x = self.feature(x)
        return x



class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)




class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

