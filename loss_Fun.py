import torch
import numpy as np
import math
import torch.nn.functional as F
import torch.nn as nn
import cv2
from torch.autograd import Variable
from config import params

def neg_loss(preds, gt):
    pos_inds = gt.eq(1.0)    
    neg_inds = gt.lt(1.0)
    pos_alpha = 100.0
    neg_weights = torch.pow(1 - gt[neg_inds], 4.0)
#    neg_weights = torch.pow(gt[neg_inds], 4.0)
#    neg_weights = 5e-3
    loss = 0
#    for pred in preds:
    pos_pred = preds[pos_inds]
 #   print(torch.unique(pos_pred))
    neg_pred = preds[neg_inds]

    pos_loss = torch.log(pos_pred+1e-12) * torch.pow(1 - pos_pred, 2) * pos_alpha
    neg_loss = torch.log(1 - neg_pred+1e-12) * torch.pow(neg_pred, 2) * neg_weights

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if pos_pred.nelement() == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

class CeLoss:
    """计算BCEloss分类"""
    def __init__(self):
        self.alpha1 = Variable(torch.tensor(1.0).cuda(), requires_grad=False)
        self.alpha2 = Variable(torch.tensor(1.0).cuda(), requires_grad=False)
        w = self.alpha1/self.alpha2
        self.loss = torch.nn.BCEWithLogitsLoss(pos_weight=w)
#        self.loss = torch.nn.BCELoss(reduction='none')
#        self.loss1 = torch.nn.BCELoss()
        self.fcloss = FocalLoss(alpha=0.75)

    def _sum(self,I, C, O):
        return torch.sum(I*C*torch.log(O+1e-12))

    def __call__(self, Cx, Ox):
        assert Ox.size() == Cx.size()
        Cx = Cx.detach()
 #       return self.fcloss(Ox, Cx)
#        return -self.alpha1*self._sum(Cx, Ox) - self.alpha2*self._sum(1-Cx, 1-Ox) 
     #   Ix = torch.rand_like(Cx)
     #   Ix[(Cx==1.0)|(Ix<0.1)] = 1.0
     #   Ix[Ix!=1.0] = 0
     #   Ix.detach_()
     #   Ft, T = torch.sum(Ix), torch.sum(Cx)
     #   self.alpha1 = (Ft-T)/100.0
     #   self.alpha2 = (T)/100.0
     #   L = -self.alpha1*self._sum(Ix, Cx, torch.sigmoid(Ox)) - self.alpha2*self._sum(Ix, 1-Cx, 1-torch.sigmoid(Ox))
     #   return L/torch.sum(Ix)
        return neg_loss(torch.sigmoid(Ox), Cx)
        return self.loss(Ox, Cx)


class DetLoss:
    """分别求两帧的分割loss"""
    def __init__(self, train_CE, target_CE):
        self.train_CE = train_CE
        self.target_CE = target_CE

    def __call__(self, labels1, labels2, out1, out2):
        return self.train_CE(labels1, out1)+self.target_CE(labels2, out2)


class FeatLoss:
    """计算两帧的回归loss"""
    def __init__(self, m):
        self.m = m
        self.triplet_loss = torch.nn.TripletMarginLoss(margin=self.m, p=2)

    def _dist(self, pred, target):
        return F.pairwise_distance(pred, target, p=2)

    def __call__(self, out1_0, out2_0, out2_1):
        return self.triplet_loss(out1_0, out2_0, out2_1)

class MYLoss:
    def __init__(self, feat_loss, det_loss, c):
        self.feat_loss = feat_loss
        self.det_loss = det_loss
        self.lambda_feat = 100
        self.lambda_det = 0.01
        self.c = 8

    def __call__(self, labels1, labels2, out1_0, out1, out2_0, out2, out2_1):
        loss1 = self.lambda_feat * self.feat_loss(out1_0, out2_0, out2_1)
        loss2 = self.lambda_det * self.det_loss(labels1, labels2, out1, out2)
        out2_0 = out2_0.detach()
        out2_1 = out2_1.detach()
#        out2.detach_()
        if not params['cal_match']:
            return loss2
        loss = loss1+loss2
#        print(loss1.item(), loss2.item())
        return loss, loss1, loss2

class FocalLoss(nn.Module):
    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',
                 ignore_lb=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_lb = ignore_lb

    def forward(self, logits, label):
        '''
        args: logits: tensor of shape (N, H, W)
        args: label: tensor of shape(N, H, W)
        '''
        # overcome ignored label
 #       ignore = label.data.cpu() == self.ignore_lb
 #       n_valid = (ignore == 0).sum()
 #       label[ignore] = 0

 #       ignore = ignore.nonzero()
 #       print(ignore)
 #       _, M = ignore.size()
 #       a, *b = ignore.chunk(M, dim=1)
 #       mask = torch.ones_like(logits)
 #       print([a, torch.arange(mask.size(1)), *b])
 #       mask[[a, torch.arange(mask.size(1)), *b]] = 0

        # compute loss
        print(torch.all(logits == label))
        probs = torch.sigmoid(logits)
#        lb_one_hot = logits.data.clone().zero_().scatter_(1, label.unsqueeze(1), 1)
        pt = torch.where(label == 1, probs, 1-probs)
        alpha = self.alpha*label + (1-self.alpha)*(1-label)
        loss = -alpha*pt*((1-pt)**self.gamma)*torch.log(pt + 1e-12)
 #       loss[mask == 0] = 0
        if self.reduction == 'mean':
            loss = loss.sum(dim=1).sum()/label.sum()
        return loss


