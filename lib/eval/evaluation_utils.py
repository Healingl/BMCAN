import numpy as np
import random
from glob import glob
import os
import SimpleITK as sitk
from lib.eval.evaluation_metics import *

def evalate_pred_and_origin_seg(origin_seg_array =None, predict_seg_array = None ):
    """

    :param origin_seg_array: (Z,X,Y)
    :param predict_seg_array: (Z,X,Y)
    :return:
    """

    eval_metric_dict = {}



    origin_patient_seg_array = origin_seg_array
    predict_patient_seg_array = predict_seg_array

    # compute the evaluation metrics
    Dice_complete = DSC_whole(predict_patient_seg_array, origin_patient_seg_array)
    Dice_enhancing = DSC_en(predict_patient_seg_array, origin_patient_seg_array)
    Dice_core = DSC_core(predict_patient_seg_array, origin_patient_seg_array)

    Sensitivity_whole = sensitivity_whole(predict_patient_seg_array, origin_patient_seg_array)
    Sensitivity_en = sensitivity_en(predict_patient_seg_array, origin_patient_seg_array)
    Sensitivity_core = sensitivity_core(predict_patient_seg_array, origin_patient_seg_array)

    Specificity_whole = specificity_whole(predict_patient_seg_array, origin_patient_seg_array)
    Specificity_en = specificity_en(predict_patient_seg_array, origin_patient_seg_array)
    Specificity_core = specificity_core(predict_patient_seg_array, origin_patient_seg_array)

    Hausdorff_whole = hausdorff_whole(predict_patient_seg_array, origin_patient_seg_array)
    Hausdorff_en = hausdorff_en(predict_patient_seg_array, origin_patient_seg_array)
    Hausdorff_core = hausdorff_core(predict_patient_seg_array, origin_patient_seg_array)

    if True:
        print("************************************************************")
        print("Dice complete tumor score : {:0.4f}".format(Dice_complete))
        print("Dice core tumor score (tt sauf vert): {:0.4f}".format(Dice_core))
        print("Dice enhancing tumor score (jaune):{:0.4f} ".format(Dice_enhancing))
        print("**********************************************")
        print("Sensitivity complete tumor score : {:0.4f}".format(Sensitivity_whole))
        print("Sensitivity core tumor score (tt sauf vert): {:0.4f}".format(Sensitivity_core))
        print("Sensitivity enhancing tumor score (jaune):{:0.4f} ".format(Sensitivity_en))
        print("***********************************************")
        print("Specificity complete tumor score : {:0.4f}".format(Specificity_whole))
        print("Specificity core tumor score (tt sauf vert): {:0.4f}".format(Specificity_core))
        print("Specificity enhancing tumor score (jaune):{:0.4f} ".format(Specificity_en))
        print("***********************************************")
        print("Hausdorff complete tumor score : {:0.4f}".format(Hausdorff_whole))
        print("Hausdorff core tumor score (tt sauf vert): {:0.4f}".format(Hausdorff_core))
        print("Hausdorff enhancing tumor score (jaune):{:0.4f} ".format(Hausdorff_en))
        print("***************************************************************\n\n")
    eval_metric_dict['Dice_complete'] = Dice_complete
    eval_metric_dict['Dice_core'] = Dice_core
    eval_metric_dict['Dice_enhancing'] = Dice_enhancing

    eval_metric_dict['Sensitivity_whole'] = Sensitivity_whole
    eval_metric_dict['Sensitivity_core'] = Sensitivity_core
    eval_metric_dict['Sensitivity_en'] = Sensitivity_en

    eval_metric_dict['Specificity_whole'] = Specificity_whole
    eval_metric_dict['Specificity_core'] = Specificity_core
    eval_metric_dict['Specificity_en'] = Specificity_en

    eval_metric_dict['Hausdorff_whole'] = Hausdorff_whole
    eval_metric_dict['Hausdorff_core'] = Hausdorff_core
    eval_metric_dict['Hausdorff_en'] = Hausdorff_en

    # Dice_WT Dice_CT Dice_ET  Sensitivity_WT Sensitivity_CT Sensitivity_ET Specificity_WT Specificity_CT Specificity_ET Hausdorff_WT Hausdorff_CT Hausdorff_ET
    return eval_metric_dict




