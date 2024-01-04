#! /usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Desc: 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lib.eval.cmda_metric import cmda_eval_metrics
from torch.optim.lr_scheduler import StepLR

from lib.utils.logging import *
from tqdm import tqdm
from lib.utils.simple_parser import Parser
import pandas as pd
import numpy as np
import cv2
import itertools
import shutil

from lib.dataloader.MMWHPairedDataset import MMWHPairedDataset
from lib.dataloader.MMWHPairedDataset import mmwh_process_img
from lib.dataloader.MMWHPairedDataset import load_nifty_volume_as_array
from lib.dataloader.MMWHPairedDataset import itensity_normalization

import random
import torch.backends.cudnn as cudnn
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm
from lib.bmcan_architecture.util.loss import VGGLoss, VGGLoss_for_trans, cross_entropy2d, entropy2d, dice_loss, ContentLoss, NormalNLLLoss
from lib.bmcan_architecture.util.utils import poly_lr_scheduler, adjust_learning_rate, save_models, load_models

import time
current_time = str(time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time())))


def validate_by_real_mr_patient(seg_model, epoch):
    """

    :param seg_model:
    :param epoch:
    :return:
    """

    origin_val_csv = pd.read_csv(model_yaml_config['all_mmwh_cmda_real_test_mr_data_csv_path'])
    mr_pred_img_dir = os.path.join(workdir,'val_real_test_mr_pred/')

    if not os.path.exists(mr_pred_img_dir):os.mkdir(mr_pred_img_dir)
    seg_model.eval()

    with torch.no_grad():
        # {'bg':0, 'tumor':1,  'cochlea':2}
        # patient_id, MeanDSC, MeanASSD, DSC_tumor, DSC_cochlea,  ASSD_tumor, ASSD_cochlea
        cdam_eval_results = []
        for idx, row in tqdm(origin_val_csv.iterrows(), total=len(origin_val_csv), ncols=50):
            # if idx > 1:
            #     break
            # input data
            current_patient_mr_path = row['img_path']
            # label
            current_patient_seg_path = row['label_path']

            current_patient_id = os.path.basename(current_patient_mr_path).replace('.nii.gz','')

            current_patient_seg_array, _, _ = load_nifty_volume_as_array(current_patient_seg_path, with_header=True, is_mmwh=True)
            fix_segmentation_map = current_patient_seg_array
            full_vol_dim = fix_segmentation_map.shape

            # read mr
            current_patient_mr_array, mr_affine, mr_header = load_nifty_volume_as_array(current_patient_mr_path,
                                                                                        with_header=True, is_mmwh=True)
            # normalization
            current_patient_mr_norm = mmwh_process_img(image_data=current_patient_mr_array, modality='mr')


            # batch_size
            slice_num = full_vol_dim[0]

            # # # # # # # #
            # visual
            # # # # # # # #

            current_patient_visual_mr_and_pred_mask_img_list = []
            current_patient_mr_255 = np.uint8(itensity_normalization(current_patient_mr_norm, norm_type='max_min') * 255)

            current_prediction_volume = np.zeros_like(current_patient_seg_array)

            for current_slice_idx in range(0, slice_num):

                # mr
                three_channel_mr_map = current_patient_mr_norm[current_slice_idx]

                # [1,1,256,256]
                feature_np_array = np.array([[three_channel_mr_map]])
                input_feature_tensor = torch.from_numpy(feature_np_array).float()
                inputs = input_feature_tensor.cuda(non_blocking=True)

                seg_input = inputs

                # predict
                _, _, pred, _, _ = seg_model(seg_input)
                pred = upsample(pred)
                pred = softmax(pred)
                pred = pred.data.max(1)[1].cpu().numpy()
                del inputs

                centre_seg_img = pred

                # # # # # # # # # # # # # #
                # # centre window end
                # # # # # # # # # # # # # #
                current_prediction_volume[current_slice_idx] = centre_seg_img

            patient_predict_seg_array = np.array(current_prediction_volume).astype(np.uint8)
            dsc_dict, assd_dict = cmda_eval_metrics(gt=fix_segmentation_map, pred=patient_predict_seg_array,
                                                      class_label_list=[1, 2, 3, 4])

            # DSC_tumor, DSC_cochlea,  ASSD_tumor, ASSD_cochlea
            # # {'bg':0, 'tumor':1,  'cochlea':2}
            # {'0':'bg', '1':'la_myo',  '2':'la_blood', '3':'lv_blood', '4':'aa'}

            DSC_la_myo = dsc_dict['la_myo']
            DSC_la_blood = dsc_dict['la_blood']
            DSC_lv_blood = dsc_dict['lv_blood']
            DSC_aa = dsc_dict['aa']

            ASSD_la_myo = assd_dict['la_myo']
            ASSD_la_blood = assd_dict['la_blood']
            ASSD_lv_blood = assd_dict['lv_blood']
            ASSD_aa = assd_dict['aa']

            cdam_eval_results.append([current_patient_id, DSC_la_myo, DSC_la_blood, DSC_lv_blood, DSC_aa, ASSD_la_myo, ASSD_la_blood, ASSD_lv_blood, ASSD_aa])


        cdam_eval_result_csv = pd.DataFrame(
            columns=['patient_id', 'DSC_la_myo', 'DSC_la_blood', 'DSC_lv_blood', 'DSC_aa', 'ASSD_la_myo', 'ASSD_la_blood', 'ASSD_lv_blood', 'ASSD_aa'],
            data=cdam_eval_results)

        mean_dsc_la_myo = round(cdam_eval_result_csv['DSC_la_myo'].mean(), 4)
        mean_dsc_la_blood = round(cdam_eval_result_csv['DSC_la_blood'].mean(), 4)
        mean_dsc_lv_blood = round(cdam_eval_result_csv['DSC_lv_blood'].mean(), 4)
        mean_dsc_aa = round(cdam_eval_result_csv['DSC_aa'].mean(), 4)

        eval_dsc_dict = {'DSC_la_myo': mean_dsc_la_myo, 'DSC_la_blood': mean_dsc_la_blood, 'DSC_lv_blood': mean_dsc_lv_blood, 'DSC_aa': mean_dsc_aa}

        mean_assd_la_myo = round(cdam_eval_result_csv['ASSD_la_myo'].mean(), 4)
        mean_assd_la_blood = round(cdam_eval_result_csv['ASSD_la_blood'].mean(), 4)
        mean_assd_lv_blood = round(cdam_eval_result_csv['ASSD_lv_blood'].mean(), 4)
        mean_assd_aa = round(cdam_eval_result_csv['ASSD_aa'].mean(), 4)

        eval_assd_dict = {'ASSD_la_myo': mean_assd_la_myo, 'ASSD_la_blood': mean_assd_la_blood, 'ASSD_lv_blood': mean_assd_lv_blood, 'ASSD_aa': mean_assd_aa}

        mean_dice_value = round((mean_dsc_la_myo + mean_dsc_la_blood + mean_dsc_lv_blood + mean_dsc_aa) / 4, 4)
        mean_assd_value = round((mean_assd_la_myo + mean_assd_la_blood + mean_assd_lv_blood + mean_assd_aa) / 4, 4)


        return mean_dice_value, mean_assd_value, eval_dsc_dict, eval_assd_dict


if __name__ == "__main__":
    model_config_path = './config/model_config/BMCAN_CT2MR_Res2Next.yaml'
    model_yaml_config = Parser(model_config_path)

    model_name = model_yaml_config['model_name']
    workdir = model_yaml_config['workdir']
    if not os.path.exists(workdir):os.makedirs(workdir)

    logger = get_logger(os.path.join(workdir, '%s.log'%(model_name)))
    shutil.copy(model_config_path,os.path.join(workdir, os.path.basename(model_config_path)))

    logger.info(str(model_yaml_config))

    gpu_list = model_yaml_config['gpus']
    train_batch_size = model_yaml_config['train_batch_size']

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # 加载数据集
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    train_dataset = MMWHPairedDataset(
        mmwh_ct_sample_data_csv_path=model_yaml_config['all_mmwh_cmda_ct_train_sample_csv_path'],
        mmwh_mr_sample_data_csv_path=model_yaml_config['all_mmwh_cmda_mr_train_sample_csv_path'],
        mode='train',
        data_num=-1,
        use_aug=model_yaml_config['use_aug'])

    val_dataset = MMWHPairedDataset(
        mmwh_ct_sample_data_csv_path=model_yaml_config['all_mmwh_cmda_ct_val_sample_csv_path'],
        mmwh_mr_sample_data_csv_path=model_yaml_config['all_mmwh_cmda_mr_val_sample_csv_path'],
        mode='val',
        data_num=-1,
        use_aug=False)

    train_loader = DataLoader(dataset=train_dataset, batch_size=model_yaml_config['train_batch_size'],
                              num_workers=0, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=model_yaml_config['val_batch_size'], num_workers=0,
                            shuffle=True)

    logger.info('train_loader num: %s' % (len(train_loader)))
    torch.cuda.set_device('cuda:{}'.format(gpu_list[0]))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # 训练开始
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    label = ['BACKGROUND', 'MYO', 'LAC', 'LVC', 'AA', ]

    GEN_IMG_DIR = os.path.join(model_yaml_config['workdir'], 'generated_imgs')
    MODEL_DIR = os.path.join(model_yaml_config['workdir'], 'model')
    if not os.path.exists(GEN_IMG_DIR):
        os.makedirs(GEN_IMG_DIR)
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    torch.manual_seed(model_yaml_config['seed'])
    torch.cuda.manual_seed(model_yaml_config['seed'])
    np.random.seed(model_yaml_config['seed'])
    random.seed(model_yaml_config['seed'])
    torch.cuda.manual_seed_all(model_yaml_config['seed'])
    torch.random.manual_seed(model_yaml_config['seed'])
    random.seed(model_yaml_config['seed'])
    cudnn.enabled = model_yaml_config['cudnn_enabled']
    cudnn.benchmark = model_yaml_config['cudnn_benchmark']


    input_size = [model_yaml_config['size_C'], model_yaml_config['size_H'], model_yaml_config['size_W']]

    # 10*2000
    num_steps = model_yaml_config['max_epoch'] * model_yaml_config['iter_val']

    # Setup Metrics
    model_dict = {}

    # Setup Model
    print('building models ...')
    from lib.bmcan_architecture.res2next import BMCANSharedEncoder
    from lib.bmcan_architecture.discriminator import SynImageDecoder, PatchGANDiscriminator
    from lib.bmcan_architecture.patchnce import PatchSampleMLP, PatchNCELoss

    seg_shared = BMCANSharedEncoder(in_dim=model_yaml_config['size_C'], n_class=model_yaml_config['n_class']).cuda()

    dclf1 = PatchGANDiscriminator(in_dim=model_yaml_config['n_class'], out_dim=1).cuda()

    dclf3 = dclf1

    dec_s = SynImageDecoder(model_yaml_config['size_C'], model_yaml_config['shared_code_channels']).cuda()
    dec_t = SynImageDecoder(model_yaml_config['size_C'], model_yaml_config['shared_code_channels']).cuda()

    dis_t = PatchGANDiscriminator(model_yaml_config['size_C'], 1).cuda()
    dis_s = PatchGANDiscriminator(model_yaml_config['size_C'], 1).cuda()

    nce_feature_dim_list = model_yaml_config['nce_feature_dim_list']

    patch_sample_f_s = PatchSampleMLP(nce_feature_dim_list=nce_feature_dim_list, use_mlp=True, init_gain=model_yaml_config['init_gain'], nc=model_yaml_config['netF_nc']).cuda()
    patch_sample_f_t = PatchSampleMLP(nce_feature_dim_list=nce_feature_dim_list, use_mlp=True, init_gain=model_yaml_config['init_gain'], nc=model_yaml_config['netF_nc']).cuda()

    model_dict['seg_shared'] = seg_shared

    model_dict['dclf1'] = dclf1
    model_dict['dclf3'] = dclf3

    model_dict['dec_s'] = dec_s
    model_dict['dec_t'] = dec_t
    model_dict['dis_s2t'] = dis_t
    model_dict['dis_t2s'] = dis_s
    model_dict['patch_sample_f_s'] = patch_sample_f_s
    model_dict['patch_sample_f_t'] = patch_sample_f_t

    seg_shared_opt = optim.Adam(seg_shared.parameters(),
                                lr=model_yaml_config['lr_seg'],
                                betas=model_yaml_config['betas']
                                )

    dclf1_opt = optim.Adam(dclf1.parameters(), lr=model_yaml_config['lr_dp'], betas=model_yaml_config['gan_betas'])
    dclf3_opt = optim.Adam(dclf3.parameters(), lr=model_yaml_config['lr_dp'], betas=model_yaml_config['gan_betas'])

    dec_s_opt = optim.Adam(dec_s.parameters(), lr=model_yaml_config['lr_rec'], betas=model_yaml_config['gan_betas'])
    dec_t_opt = optim.Adam(dec_t.parameters(), lr=model_yaml_config['lr_rec'], betas=model_yaml_config['gan_betas'])

    dis_t_opt = optim.Adam(dis_t.parameters(), lr=model_yaml_config['lr_dis'], betas=model_yaml_config['gan_betas'])
    dis_s_opt = optim.Adam(dis_s.parameters(), lr=model_yaml_config['lr_dis'], betas=model_yaml_config['gan_betas'])


    patch_sample_f_s_opt = optim.Adam(patch_sample_f_s.parameters(), lr=model_yaml_config['lr_rec'], betas=model_yaml_config['gan_betas'])
    patch_sample_f_t_opt = optim.Adam(patch_sample_f_t.parameters(), lr=model_yaml_config['lr_rec'], betas=model_yaml_config['gan_betas'])

    # Optimizer list for quickly adjusting learning rate
    seg_opt_list = [seg_shared_opt]
    dclf_opt_list = [dclf1_opt, dclf3_opt]
    rec_opt_list = [dec_s_opt, dec_t_opt]
    dis_opt_list = [dis_t_opt, dis_s_opt]
    patch_mlp_opt_list = [patch_sample_f_s_opt, patch_sample_f_t_opt]

    if not model_yaml_config['resume'] == 0:
        load_models(model_dict, os.path.join(MODEL_DIR, 'weight_%06d' % model_yaml_config['resume']))
        print('Model loaded from %s' % os.path.join(MODEL_DIR, 'weight_%06d' % model_yaml_config['resume']))

    dc_loss = dice_loss
    l1_loss = nn.L1Loss(reduction='mean').cuda()
    mse_loss = nn.MSELoss(reduction='mean').cuda()
    bce_loss = nn.BCEWithLogitsLoss().cuda()
    sg_loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')

    recon_loss = l1_loss
    dp_loss = mse_loss
    di_loss = mse_loss

    from lib.bmcan_architecture.anatomy_loss import Cor_CoeLoss
    correlation_loss = Cor_CoeLoss


    criterionNCE = []
    for nce_feature_dim in nce_feature_dim_list:
        criterionNCE.append(PatchNCELoss(model_yaml_config['nce_T']))

    upsample = nn.Upsample(size=[model_yaml_config['size_H'],
                                 model_yaml_config['size_W']],
                           mode='bilinear',
                           align_corners=True)

    softmax = lambda x: F.softmax(x, dim=1)
    entropy = lambda prob: -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(prob.shape[1])

    true_label = 1
    fake_label = 0


    dclf1.train()
    dclf3.train()

    seg_shared.train()
    dec_s.train()
    dec_t.train()
    dis_t.train()
    dis_s.train()
    patch_sample_f_t.train()
    patch_sample_f_s.train()

    best_dice = 0
    best_iter = 0

    train_ct_and_mr_iter = enumerate(train_loader)

    aux_constrained_loss_dict = {}

    # # # # # # # # # # # # # # # # # # # # # # # # #
    # # Training
    # # # # # # # # # # # # # # # # # # # # # # # # #

    for i_iter in tqdm(range(model_yaml_config['resume'], num_steps + 1), total=num_steps - model_yaml_config['resume']):

        # Evaluate
        if i_iter % model_yaml_config['iter_val'] == 0 and i_iter != 0:

            epoch = i_iter // model_yaml_config['iter_val']

            logger.info('>>>>' * 30)
            logger.info('Evaluate MR Patient:')
            mean_dice_value, mean_assd_value, eval_dsc_dict, eval_assd_dict = validate_by_real_mr_patient(seg_model=seg_shared,

                                                                                                          epoch=epoch)

            logger.info('mean_dice_value: %s, mean_assd_value: %s' % (mean_dice_value, mean_assd_value))
            logger.info('eval_dsc_dict: %s' % (str(eval_dsc_dict)))
            logger.info('eval_assd_dict: %s' % (str(eval_assd_dict)))

            if mean_dice_value > best_dice:
                best_iter = i_iter
                best_dice = mean_dice_value
                save_models(model_dict, MODEL_DIR + '/weight_%s_meanDC_%s/'%(i_iter,mean_dice_value))

            logger.info('Best Iter %06d : Dice %.2f ' % (best_iter, best_dice))
            logger.info('>>>>' * 30)
            if i_iter == num_steps:
                break


        seg_shared.train()

        if model_yaml_config['lr_schedule']:
            adjust_learning_rate(seg_opt_list, base_lr=model_yaml_config['lr_seg'], i_iter=i_iter, max_iter=num_steps,
                                 power=model_yaml_config['power'])
            adjust_learning_rate(dclf_opt_list, base_lr=model_yaml_config['lr_dp'], i_iter=i_iter, max_iter=num_steps,
                                 power=model_yaml_config['power'])

            adjust_learning_rate(rec_opt_list, base_lr=model_yaml_config['lr_rec'], i_iter=i_iter, max_iter=num_steps,
                                 power=model_yaml_config['power'])
            adjust_learning_rate(patch_mlp_opt_list, base_lr=model_yaml_config['lr_rec'], i_iter=i_iter,
                                 max_iter=num_steps,
                                 power=model_yaml_config['power'])

            adjust_learning_rate(dis_opt_list, base_lr=model_yaml_config['lr_dis'], i_iter=i_iter, max_iter=num_steps,
                                 power=model_yaml_config['power'])

        # # # # # # # # # # # # # # # # # # # # # # # # #
        # # Train tensor getting
        # # # # # # # # # # # # # # # # # # # # # # # # #

        train_ct_input, train_ct_label, train_mr_input, train_mr_label = next(iter(train_loader))

        # ct2mr
        source_data, source_label, target_data, _ = train_ct_input.cuda(non_blocking=True), train_ct_label.cuda(non_blocking=True), train_mr_input.cuda(non_blocking=True), train_mr_label.cuda(non_blocking=True)

        # source
        sdatav = source_data.cuda()
        slabelv = source_label.type(torch.LongTensor).cuda()

        # target
        tdatav = target_data.cuda()

        # # # # # # # # # # # # # # # # # # # # # # # # #

        # # # # # # # # # # # # # # # # # # # # # # # # #
        # # Segmentation loss cal
        # # # # # # # # # # # # # # # # # # # # # # # # #
        # seg_shared forwarding, output: layer0 output, n_class channels pred1, n_class channels pred2, n_dim code_s_common
        low_s, s_pred1, s_pred2, code_s_common, feat_true_s_list = seg_shared(sdatav)
        low_t, t_pred1, t_pred2, code_t_common, feat_true_t_list = seg_shared(tdatav)

        # upsample
        s_pred1 = upsample(s_pred1)
        s_pred2 = upsample(s_pred2)
        t_pred1 = upsample(t_pred1)
        t_pred2 = upsample(t_pred2)

        # seg output
        # source pred
        s_pred = s_pred2
        # target pred
        t_pred = t_pred2

        # ==== segmentation loss ====
        loss_s_ce = sg_loss(s_pred, slabelv)
        loss_s_dc = dice_loss(s_pred, slabelv)
        loss_sup_seg = loss_s_ce + loss_s_dc
        s_sup_dice_loss = model_yaml_config['lambda_seg_sup_s'] * loss_sup_seg
        total_loss = s_sup_dice_loss
        aux_constrained_loss_dict['s_sup_dice_loss'] = s_sup_dice_loss
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # aux deep supervision
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        loss_sup_s_ce1 = sg_loss(s_pred1, slabelv)
        loss_sup_s_dc1 = dice_loss(s_pred1, slabelv)

        loss_aux_seg_loss = model_yaml_config['lambda_aux'] * (loss_sup_s_ce1 + loss_sup_s_dc1)
        total_loss += loss_aux_seg_loss


        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # # decode within domain and calculate reconstruct loss from code_common
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # decoder source: code_s_common ->  rec_s
        rec_s = dec_s(code_s_common)
        rec_t = dec_t(code_t_common)

        # ==== reconstruction loss ====
        if model_yaml_config['lambda_rec'] > 0:

            # print('lambda_rec', model_yaml_config['lambda_rec'])
            loss_rec_s = recon_loss(rec_s, sdatav)
            loss_rec_t = recon_loss(rec_t, tdatav)
            loss_rec_self = loss_rec_s + loss_rec_t

            total_loss += model_yaml_config['lambda_rec'] * loss_rec_self

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # # decode cross domain and calculate discriminator loss from code_common
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #



        # ==== image translation loss ====
        if model_yaml_config['lambda_adv_i'] > 0:
            rec_s2t = dec_t(code_s_common)
            rec_t2s = dec_s(code_t_common)



            # train image discriminator -> LSGAN
            for p in dis_t.parameters():
                p.requires_grad = True
            for p in dis_s.parameters():
                p.requires_grad = True

            # ===== dis_s2t =====
            if i_iter % 1 == 0:
                prob_dis_t_real1 = dis_t(tdatav)
                prob_dis_t_fake1 = dis_t(rec_s2t.detach())

                #
                loss_d_t = 0.5 * di_loss(prob_dis_t_real1, torch.empty(prob_dis_t_real1.shape).fill_(true_label).cuda()) + \
                           0.5 * di_loss(prob_dis_t_fake1, torch.empty(prob_dis_t_fake1.shape).fill_(fake_label).cuda())


                dis_t_opt.zero_grad()
                loss_d_t.backward()
                dis_t_opt.step()


            # ===== dis_t2s =====
            if i_iter % 1 == 0:
                prob_dis_s_real1 = dis_s(sdatav)
                prob_dis_s_fake1 = dis_s(rec_t2s.detach())

                loss_d_s = 0.5 * di_loss(prob_dis_s_real1,torch.empty(prob_dis_s_real1.shape).fill_(true_label).cuda()) + \
                           0.5 * di_loss(prob_dis_s_fake1, torch.empty(prob_dis_s_fake1.shape).fill_(fake_label).cuda())


                dis_s_opt.zero_grad()
                loss_d_s.backward()
                dis_s_opt.step()

            for p in dis_t.parameters():
                p.requires_grad = False
            for p in dis_s.parameters():
                p.requires_grad = False

            prob_dis_t_fake2 = dis_t(rec_s2t)
            loss_gen_s2t = di_loss(prob_dis_t_fake2, torch.empty(prob_dis_t_fake2.shape).fill_(true_label).cuda())

            prob_dis_s_fake2 = dis_s(rec_t2s)
            loss_gen_t2s = di_loss(prob_dis_s_fake2, torch.empty(prob_dis_s_fake2.shape).fill_(true_label).cuda())

            loss_image_translation = loss_gen_s2t + loss_gen_t2s
            total_loss += model_yaml_config['lambda_adv_i'] * loss_image_translation

            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            # # cycle seg loss
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            aux_constrained_loss_dict['loss_bmcl_nce'] = torch.tensor(0)
            aux_constrained_loss_dict['loss_anatomy_constrained'] = torch.tensor(0)

            # When to start using translated labels, it should be discussed
            if model_yaml_config['lambda_s2t_seg'] > 0 and i_iter > model_yaml_config['start_s2t_seg']:


                # break
                # check if we have to detach the rec_s2t images
                # s->t
                _, s2t_pred1, s2t_pred2, code_s2t_common, feat_fake_s2t_list = seg_shared(rec_s2t.detach())
                _, t2s_pred1, t2s_pred2, code_t2s_common, feat_fake_t2s_list = seg_shared(rec_t2s.detach())

                s2t_pred1 = upsample(s2t_pred1)
                s2t_pred2 = upsample(s2t_pred2)

                t2s_pred1 = upsample(t2s_pred1)
                t2s_pred2 = upsample(t2s_pred2)

                s2t_pred = s2t_pred2
                t2s_pred = t2s_pred2

                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
                # deep supervision
                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

                loss_sup_s2t_ce1 = sg_loss(s2t_pred1, slabelv)
                loss_sup_s2t_ce2 = sg_loss(s2t_pred2, slabelv)
                loss_sup_s2t_ce = model_yaml_config['lambda_aux'] * loss_sup_s2t_ce1 + loss_sup_s2t_ce2

                loss_sup_s2t_dc1 = dice_loss(s2t_pred1, slabelv)
                loss_sup_s2t_dc2 = dice_loss(s2t_pred2, slabelv)
                loss_sup_s2t_dc = model_yaml_config['lambda_aux'] * loss_sup_s2t_dc1 + loss_sup_s2t_dc2

                loss_sup_s2t_sg = loss_sup_s2t_ce + loss_sup_s2t_dc
                total_loss += model_yaml_config['lambda_s2t_seg'] * loss_sup_s2t_sg


                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
                # # Calculate Multilayer Noise Contrastive Estimation
                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
                # anatomy structure loss
                loss_CC = Cor_CoeLoss(rec_s2t.detach(), sdatav) + Cor_CoeLoss(rec_t2s.detach(), tdatav)
                loss_anatomy_constrained = model_yaml_config['lambda_ana_consis'] * loss_CC
                aux_constrained_loss_dict['loss_anatomy_constrained'] = loss_anatomy_constrained
                total_loss += loss_anatomy_constrained

                loss_bmcl_nce = 0

                # ==== Fake NCE loss ====
                if model_yaml_config['lambda_NCE'] > 0:
                    n_layers = len(nce_feature_dim_list)

                    # calculate_NCE_loss1: real_a -> fake_b
                    feat_q = feat_fake_s2t_list
                    feat_k = feat_true_s_list


                    feat_k_pool, sample_ids = patch_sample_f_s(feat_k, model_yaml_config.num_patches, None)
                    feat_q_pool, _ = patch_sample_f_t(feat_q, model_yaml_config.num_patches, sample_ids)
                    total_nce_loss_3 = 0.0
                    for f_q, f_k, crit, _ in zip(feat_q_pool, feat_k_pool, criterionNCE, nce_feature_dim_list):
                        loss = crit(f_q, f_k)
                        total_nce_loss_3 += loss.mean()
                    loss_NCE3 = total_nce_loss_3 / n_layers

                    # calculate_NCE_loss1: real_b -> fake_a (patch_sample_f_t -> patch_sample_f_s)
                    feat_q = feat_fake_t2s_list
                    feat_k = feat_true_t_list

                    feat_k_pool, sample_ids = patch_sample_f_t(feat_k, model_yaml_config.num_patches, None)
                    feat_q_pool, _ = patch_sample_f_s(feat_q, model_yaml_config.num_patches, sample_ids)
                    total_nce_loss_4 = 0.0
                    for f_q, f_k, crit, _ in zip(feat_q_pool, feat_k_pool, criterionNCE, nce_feature_dim_list):
                        loss = crit(f_q, f_k)
                        total_nce_loss_4 += loss.mean()
                    loss_NCE4 = total_nce_loss_4 / n_layers

                    loss_bmcl_nce = model_yaml_config['lambda_NCE'] *(loss_NCE3 + loss_NCE4)
                    aux_constrained_loss_dict['loss_bmcl_nce'] = loss_bmcl_nce
                    total_loss +=  loss_bmcl_nce


        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # # seg adv loss
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # ==== source domain confusion loss ====
        if model_yaml_config['lambda_adv_p'] > 0:

            # first true source img and true target image pred mask output for getting adversarial loss
            # train Domain classifier
            # ===== dclf1 =====
            for p in dclf1.parameters():
                p.requires_grad = True


            dclf1_d_real = s_pred.detach()
            dclf1_d_fake = t_pred.detach()

            if model_yaml_config['adv_p_smooth'] == 'softmax':
                dclf1_d_real = softmax(dclf1_d_real)
                dclf1_d_fake = softmax(dclf1_d_fake)

            if model_yaml_config['adv_p_smooth'] == 'entropy':
                dclf1_d_real = entropy(softmax(dclf1_d_real))
                dclf1_d_fake = entropy(softmax(dclf1_d_fake))

            prob_dclf1_d_real = dclf1(dclf1_d_real)
            prob_dclf1_d_fake = dclf1(dclf1_d_fake)

            loss_d_dclf1 = 0.5 * dp_loss(prob_dclf1_d_real, torch.empty(prob_dclf1_d_real.shape).fill_(true_label).cuda()) \
                           + 0.5 * dp_loss(prob_dclf1_d_fake, torch.empty(prob_dclf1_d_fake.shape).fill_(fake_label).cuda())
            loss_d_dclf_sum1 = loss_d_dclf1

            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            # adv seg loss for fake imgs
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            if model_yaml_config['lambda_s2t_seg'] > 0 and model_yaml_config['lambda_adv_tp'] > 0 and i_iter > model_yaml_config['start_s2t_seg']:
                for p in dclf3.parameters():
                    p.requires_grad = True

                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
                # first fake source img and fake target image pred mask output for getting adversarial loss
                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
                dclf3_d_real = s2t_pred.detach()
                dclf3_d_fake = t2s_pred.detach()

                if model_yaml_config['adv_p_smooth'] == 'softmax':
                    dclf3_d_real = softmax(dclf3_d_real)
                    dclf3_d_fake = softmax(dclf3_d_fake)
                if model_yaml_config['adv_p_smooth'] == 'entropy':
                    dclf3_d_real = entropy(softmax(dclf3_d_real))
                    dclf3_d_fake = entropy(softmax(dclf3_d_fake))


                prob_dclf3_d_real = dclf3(dclf3_d_real)
                prob_dclf3_d_fake = dclf3(dclf3_d_fake)

                loss_d_dclf3 = 0.5 * dp_loss(prob_dclf3_d_real, torch.empty(prob_dclf3_d_real.shape).fill_(
                    true_label).cuda()) + 0.5 * dp_loss(prob_dclf3_d_fake,
                                                        torch.empty(prob_dclf3_d_fake.shape).fill_(
                                                            fake_label).cuda())

                loss_d_dclf_sum1 += loss_d_dclf3


            dclf1_opt.zero_grad()
            loss_d_dclf_sum1.backward()
            dclf1_opt.step()

            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            # true target pred mask similarity loss (for unlabeled target feature similarity for short distance? it seems like adv seg label loss)
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            for p in dclf1.parameters():
                p.requires_grad = False

            dclf1_g_fake = t_pred


            if model_yaml_config['adv_p_smooth'] == 'softmax':
                dclf1_g_fake = softmax(dclf1_g_fake)
            if model_yaml_config['adv_p_smooth'] == 'entropy':
                dclf1_g_fake = entropy(softmax(dclf1_g_fake))

            prob_dclf1_g_fake = dclf1(dclf1_g_fake)

            loss_feat_similarity = dp_loss(prob_dclf1_g_fake,
                                           torch.empty(prob_dclf1_g_fake.shape).fill_(true_label).cuda())

            total_loss += model_yaml_config['lambda_adv_p'] * loss_feat_similarity

            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            # fake source pred mask adv loss (feature similarity for short distance? it seems like adv seg label loss)
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            if model_yaml_config['lambda_s2t_seg'] > 0 and model_yaml_config['lambda_adv_tp'] > 0 and i_iter > model_yaml_config['start_s2t_seg']:
                for p in dclf3.parameters():
                    p.requires_grad = False

                dclf3_g_fake = t2s_pred

                if model_yaml_config['adv_p_smooth'] == 'softmax':
                    dclf3_g_fake = softmax(dclf3_g_fake)
                if model_yaml_config['adv_p_smooth'] == 'entropy':
                    dclf3_g_fake = entropy(softmax(dclf3_g_fake))

                prob_dclf3_g_fake = dclf3(dclf3_g_fake)
                loss_feat_similarity_new = dp_loss(prob_dclf3_g_fake,
                                                torch.empty(prob_dclf3_g_fake.shape).fill_(true_label).cuda())

                total_loss += model_yaml_config['lambda_adv_tp'] * loss_feat_similarity_new

        seg_shared_opt.zero_grad()
        dec_s_opt.zero_grad()
        dec_t_opt.zero_grad()
        patch_sample_f_s_opt.zero_grad()
        patch_sample_f_t_opt.zero_grad()

        total_loss.backward()

        patch_sample_f_s_opt.step()
        patch_sample_f_t_opt.step()
        seg_shared_opt.step()
        dec_s_opt.step()
        dec_t_opt.step()

        seg_lr = seg_shared_opt.param_groups[0]['lr']
        logger.info('[iter %d / %d], [seg_lr: %.7f], [loss all: %.4f], [loss dc: %.4f], [loss nce: %.4f], [loss anatomy: %.4f]' % (num_steps, i_iter,
                                                                                                                                   seg_lr, total_loss.item(),
                                                                                                                                   round(aux_constrained_loss_dict['s_sup_dice_loss'].item(),4),
                                                                                                                                   round(aux_constrained_loss_dict['loss_bmcl_nce'].item(), 4),
                                                                                                                                   round(aux_constrained_loss_dict['loss_anatomy_constrained'].item(), 4)))


