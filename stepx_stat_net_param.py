#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# @Project: bmcan_standard_code
# @IDE: PyCharm
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Time: 2022/6/10
# @Desc: 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from lib.utils.simple_parser import Parser

model_config_path = './config/model_config/BMCAN_MR2CT_Res2Next.yaml'
model_yaml_config = Parser(model_config_path)

from lib.bmcan_architecture.res2next import BMCANOnlySharedEncoder
from lib.bmcan_architecture.res2next import BMCANSharedEncoder
from lib.bmcan_architecture.discriminator import SynImageDecoder, PatchGANDiscriminator
from lib.bmcan_architecture.patchnce import PatchSampleMLP, PatchNCELoss
from lib.bmcan_architecture.layers.utils import count_param

if __name__ == '__main__':

    # Shared Encoder
    seg_shared_and_decoder = BMCANSharedEncoder(in_dim=model_yaml_config['size_C'], n_class=model_yaml_config['n_class'])
    seg_shared_and_decoder_param = count_param(seg_shared_and_decoder)/ 1e6

    seg_dis = PatchGANDiscriminator(in_dim=model_yaml_config['n_class'], out_dim=1)
    seg_dis_param = count_param(seg_dis) / 1e6

    dec_s = SynImageDecoder(model_yaml_config['size_C'], model_yaml_config['shared_code_channels'])
    dec_t = SynImageDecoder(model_yaml_config['size_C'], model_yaml_config['shared_code_channels'])

    dec_s_param = count_param(dec_s) / 1e6
    dec_t_param = count_param(dec_t) / 1e6

    dis_t = PatchGANDiscriminator(model_yaml_config['size_C'], 1)
    dis_s = PatchGANDiscriminator(model_yaml_config['size_C'], 1)

    dis_t_param = count_param(dis_t) / 1e6
    dis_s_param = count_param(dis_s) / 1e6

    nce_feature_dim_list = model_yaml_config['nce_feature_dim_list']

    patch_sample_f_s = PatchSampleMLP(nce_feature_dim_list=nce_feature_dim_list, use_mlp=True, init_gain=model_yaml_config['init_gain'], nc=model_yaml_config['netF_nc'])
    patch_sample_f_t = PatchSampleMLP(nce_feature_dim_list=nce_feature_dim_list, use_mlp=True, init_gain=model_yaml_config['init_gain'], nc=model_yaml_config['netF_nc'])

    patch_sample_f_s_param = count_param(patch_sample_f_s) / 1e6
    patch_sample_f_t_param = count_param(patch_sample_f_t) / 1e6

    all_param = seg_shared_and_decoder_param + dec_s_param + dec_t_param + seg_dis_param + dis_t_param + dis_s_param + patch_sample_f_s_param + patch_sample_f_t_param

    print('all_param totoal parameters: %.2fM ' % (all_param))
