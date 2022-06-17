# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np
# label = ['road',
#          'sidewalk',
#          'building',
#          'wall',
#          'fence',
#          'pole',
#          'light',
#          'sign',
#          'vegetation',
#          'terrain',
#          'sky',
#          'person',
#          'rider',
#          'car',
#          'truck',
#          'bus',
#          'train',
#          'motorcycle',
#          'bycycle']

label = ['background',
         'lv',
         'la',
         'myo',
         'aa',
         ]

class runningScore(object):

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.list_hist = []

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        # print(label_trues.shape, label_preds.shape)
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        # acc = np.diag(hist).sum() / hist.sum()
        # acc_cls = np.diag(hist) / hist.sum(axis=1)
        # acc_cls = np.nanmean(acc_cls[1:])
        # iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))

        # mean_iu = np.nanmean(iu[1:])

        # freq = hist.sum(axis=1) / hist.sum()
        # fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        hist = np.stack(self.list_hist)
        dice = (2 * np.diagonal(hist, 0, 1, 2) + 0.00001) / (hist.sum(axis=1) + hist.sum(axis=2) + 0.00001)
        dice = np.mean(dice, axis=0)
        # for id in range(5):
        #     print('===>' + label[id] + ':' + str(dice[id]))
        cls_dice = dict(zip(label, dice))
        self.list_hist = []
        return {
            'Dice': dice,
               }

    def reset(self):
        self.list_hist.append(self.confusion_matrix)
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


def fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
    return hist
#返回单个dice值
def dice_eval(label_pred, label_true, n_class):
    hist = fast_hist(label_true, label_pred, n_class)   #;print(hist)
    union = hist.sum(axis=0) + hist.sum(axis=1)
    dice = (2 * np.diag(hist) + 1e-8) / (union + 1e-8)
    dice[union == 0] = np.nan
    return dice
