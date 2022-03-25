import copy

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CosineSimilarity
from torchvision import models

eps = 1e-8

class CosineEmbeddingLoss(nn.Module):
    r"""Creates a criterion that measures the loss given input tensors
    :math:`x_1`, :math:`x_2` and a `Tensor` label :math:`y` with values 1 or -1.
    This is used for measuring whether two inputs are similar or dissimilar,
    using the cosine distance, and is typically used for learning nonlinear
    embeddings or semi-supervised learning.

    The loss function for each sample is:

    .. math::
        \text{loss}(x, y) =
        \begin{cases}
        1 - \cos(x_1, x_2), & \text{if } y = 1 \\
        \max(0, \cos(x_1, x_2) - \text{margin}), & \text{if } y = -1
        \end{cases}

    Args:
        margin (float, optional): Should be a number from :math:`-1` to :math:`1`,
            :math:`0` to :math:`0.5` is suggested. If :attr:`margin` is missing, the
            default value is :math:`0`.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
    """
    __constants__ = ['margin', 'reduction']

    def __init__(self, margin=None, reduction='mean'):
        super(CosineEmbeddingLoss, self).__init__()
        self.reduction = reduction
        self.margin = torch.zeros(1).cuda() if margin is not None else margin

    def forward(self, input1, target, mu, mask=None):
        # print(input1.size(), target.size(), mu.size())
        cos_similarity = cosine_similarity(input1, mu)  # ;print(cos_similarity.min(), cos_similarity.max())
        B, C, H, W = cos_similarity.size()
        margin_flatten = self.margin.contiguous().view(1, -1, 1, 1).expand(B, C, H, W).view(-1)
        target_flatten = target.contiguous().view(-1)
        cos_similarity_flatten = cos_similarity.contiguous().view(-1)

        # print(cos_similarity.size(), target.size(), mu.size(), self.margin.size())
        loss = target_flatten * (1 - cos_similarity_flatten) + (1 - target_flatten) * F.relu(
            cos_similarity_flatten - margin_flatten)
        if mask is not None:
            mask_flatten = mask.view(B, -1, H, W).expand(B, C, H, W).contiguous().view(-1).type(torch.float)
            loss *= mask_flatten
        if self.reduction == 'mean':
            if mask is not None:
                loss = loss.sum() / (mask > 0).sum()
            else:
                loss = torch.mean(loss)

        return loss

class ContentLoss(nn.Module):
    def __init__(self, net):
        super(ContentLoss, self).__init__()
        self.net = copy.deepcopy(net)
        for p1, p2 in zip(net.parameters(), self.net.parameters()):
            p2.data = p1.data
            p2.requires_grad = False

        self.layer0 = self.net.layer0
        self.layer1 = self.net.layer1
        self.layer2 = self.net.layer2
        self.layer3 = self.net.layer3
        self.layer4 = self.net.layer4

        self.layer5 = self.net.final1
        self.layer6 = self.net.final2
        if torch.cuda.is_available():
            self.net = self.net.cuda()
        self.criterion = nn.L1Loss()
        # self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)

    def forward(self, x, y, weights=[1.0 / 64, 1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0 / 2, 1.0]):
        bs = x.size(0)
        while x.size()[3] > 1024:
            x, y = self.downsample(x), self.downsample(y)
        loss = 0
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x5 = self.layer5(x3)
        x4 = self.layer4(x3)
        x6 = self.layer6(x4)
        fx = [x0, x1, x2, x3, x4, x5, x6]
        

        y0 = self.layer0(y)
        y1 = self.layer1(y0)
        y2 = self.layer2(y1)
        y3 = self.layer3(y2)
        y5 = self.layer5(y3)
        y4 = self.layer4(y3)
        y6 = self.layer6(y4)
        fy = [y0, y1, y2, y3, y4, y5, y6]

        for i in range(0, 7):
            if i > 2:
                loss += weights[i] * self.criterion(fx[i], fy[i])
        return loss

class VGGLoss(nn.Module):
    def __init__(self, gpu_id=0):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        if torch.cuda.is_available():
            self.vgg = self.vgg.cuda()
        self.criterion = nn.L1Loss()
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)

    def forward(self, x, y, weights=[1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]):
        x = x.expand(-1, 3, -1, -1)
        y = y.expand(-1, 3, -1, -1)
        bs = x.size(0)
        while x.size()[3] > 1024:
            x, y = self.downsample(x), self.downsample(y)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class VGGLoss_for_trans(nn.Module):
    def __init__(self, gpu_id=0):
        super(VGGLoss_for_trans, self).__init__()
        self.vgg = Vgg19()
        if torch.cuda.is_available():
            self.vgg = self.vgg.cuda()
        self.criterion = nn.L1Loss()
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)

    def forward(self, trans_img, struct_img, texture_img, alpha=1.0,
                weights=[1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]):
        while trans_img.size()[3] > 1024:
            trans_img, struct_img, texture_img = self.downsample(trans_img), self.downsample(
                struct_img), self.downsample(texture_img)
        trans_vgg, struct_vgg, texture_vgg = self.vgg(trans_img), self.vgg(struct_img), self.vgg(texture_img)
        loss = 0
        for i in range(len(trans_vgg)):
            if i < 3:
                x_feat_mean = trans_vgg[i].view(trans_vgg[i].size(0), trans_vgg[i].size(1), -1).mean(2)
                y_feat_mean = alpha * texture_vgg[i].view(texture_vgg[i].size(0), texture_vgg[i].size(1), -1).mean(2) \
                              + (1 - alpha) * struct_vgg[i].view(struct_vgg[i].size(0), struct_vgg[i].size(1), -1).mean(2)
                loss += self.criterion(x_feat_mean, y_feat_mean.detach())
            else:
                loss += weights[i] * self.criterion(trans_vgg[i], struct_vgg[i].detach())
        return loss


def cross_entropy2d(input, target, weight=None, size_average=True, mask=None):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n * h * w, 1).repeat(1, c) >= 0]
    log_p = log_p.view(-1, c)
    if mask is not None:
        target *= mask.type(torch.long)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, ignore_index=255,
                      weight=weight, reduction='sum')
    if size_average:
        loss /= mask.data.sum()
    return loss


def dice_loss(input, target, p=2, ignore_index=-100):
    n, c, h, w = input.size()
    prob = F.softmax(input, dim=1)
    prob_flatten = prob.permute(0, 2, 3, 1).contiguous().view(-1, c)#;print(prob.shape)

    target_flatten = target.view(n * h * w, 1)
    mask = target_flatten != ignore_index
    target_flatten = target_flatten[mask].view(-1, 1)#;print(target.shape)

    prob_flatten = prob_flatten[mask.repeat(1, c)]
    prob_flatten = prob_flatten.contiguous().view(-1, c)

    target_one_hot = torch.scatter(torch.zeros_like(prob_flatten), 1, target_flatten, 1.0)#;print(target.sum(0))
    prob_flatten = prob_flatten[:, 1:]
    target_one_hot = target_one_hot[:, 1:]
    dc = dice(prob_flatten, target_one_hot, p)#;print(dc)
    return 1.0 - dc.mean()


def dice_loss_1(input, target, p=2):
    n, c, h, w = input.size()
    prob = F.softmax(input, dim=1)
    prob_flatten = prob.permute(0, 2, 3, 1).contiguous().view(-1, c)#;print(prob.shape)
    true_prob = F.softmax(target, dim=1)
    target_flatten = true_prob.permute(0, 2, 3, 1).contiguous().view(-1, c)
    prob_flatten = prob_flatten[:, 1:]
    target_flatten = target_flatten[:, 1:]
    dc = dice(prob_flatten, target_flatten, p)#;print(dc)
    return 1.0 - dc.mean()


def dice(y, target, p=2):
    intersection = torch.sum(y * target, dim=0)
    union = y.pow(p).sum(0) + target.pow(p).sum(0)
    return 2 * intersection / (union + eps)



def entropy2d(input, size_average=True):
    n, c, h, w = input.size()
    p = F.softmax(input, dim=1)  # ;print(p.shape)

    log_p = torch.log2(p + 1e-30)
    loss = -torch.mul(p, log_p) / np.log2(c)
    loss = loss.sum(1)  # ;print(loss.shape)
    if size_average:
        loss = loss.mean()
    else:
        loss = loss.sum()
    return loss

def entropy_metric(input1, input2, threshold=-0.2, size_average=True):
    n, c, h, w = input1.size()
    p = F.softmax(input1, dim=1)  # ;print(p.shape)
    log_p = torch.log(p)
    ent1 = -(p * log_p).sum(1)

    p = F.softmax(input2, dim=1)  # ;print(p.shape)
    log_p = torch.log(p)
    ent2 = -(p * log_p).sum(1)
    loss = ent1 - ent2  # ;print(loss.shape)
    mask = (loss.detach() < threshold).type(torch.float)
    loss = loss * mask
    if size_average:
        loss = loss.mean()
    else:
        loss = loss.sum()
    return loss


def myL1Loss(source, target):
    return torch.mean(torch.abs(source - target))


def cosine_similarity(t, center):
    norm_t = t / torch.norm(t, 2, 1, keepdim=True)
    norm_c = center / torch.norm(center, 2, 1, keepdim=True)
    similarity = torch.matmul(norm_t.permute((0, 2, 3, 1)), norm_c.transpose(0, 1)).permute((0, 3, 1, 2))
    similarity = 0.5 + similarity * 0.5
    return similarity


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class NormalNLLLoss:
    """
    Calculate the negative log likelihood
    of normal distribution.
    This needs to be minimised.
    Treating Q(cj | x) as a factored Gaussian.
    """

    def __call__(self, x, mu, var):
        var = torch.exp(var)
        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - (x - mu).pow(2).div(var.mul(2.0) + 1e-6)

        # print(var.mul(2 * np.pi), )
        nll = -(logli.mean())

        return nll