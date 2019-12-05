import torch
import heapq
import torch.nn.functional as F
import cv2
import numpy as np
import time
from config import params
from network.GCNv2 import GCNnet, later_deal
from torch.utils.data import DataLoader, Dataset
from data import MYdata
from torch import nn, optim
from loss_Fun import *
import os


def Xor(x, y):
    return torch.sum(x^y)

def caculate_loss(batch,labels1, labels2, desc1, desc2, det1, det2):
    indice1 = torch.tensor(batch).cuda()
    start = time.time()
    out1, out2 = det1, det2
    out1_0, x1, y1 = later_deal(labels1[batch], torch.index_select(desc1, index=indice1, dim=0))
    if params['cal_match']:
        out2_0, x2, y2 = later_deal(labels2[batch], torch.index_select(desc2, index=indice1, dim=0))
        out1_0, out2_0 = out1_0.permute((1,0)), out2_0.permute((1,0))
        start = time.time()
        k = 3
        """对于每个特征，在参考帧中找出前k个与之一范距离最相近的特征，如果位置距离太远则作为负样本"""
        if params['random drop']>0:
            rand_array = torch.rand(out1_0.size()[0])
            index_loc = torch.nonzero(rand_array>params['random drop']).view(-1).cuda()
            out1_0 = torch.index_select(out1_0, index=index_loc, dim=0)
            out2_0 = torch.index_select(out2_0, index=index_loc, dim=0)
            index_loc = index_loc.cpu()
            x1 = torch.index_select(x1, index=index_loc, dim=0)
            y1 = torch.index_select(y1, index=index_loc, dim=0)
            x2 = torch.index_select(x2, index=index_loc, dim=0)
            y2 = torch.index_select(y2, index=index_loc, dim=0)
        
#        l = out1_0.size()[0]
#        T = out1_0.unsqueeze(-1).repeat(1,1,l)
#        R = out2_0.unsqueeze(-1).permute(-1,1,0).repeat(l,1,1)
#        ADDMatrix = torch.sum(torch.abs(T+R), 0)
        MultiMatrix = out2_0@torch.t(out1_0)
        sort_index = MultiMatrix.sort(0, descending = True).indices[:k]
        out1_0.x = x1
        out1_0.y = y1
        out2_0.x = x2
        out2_0.y = y2
        sort_x2, sort_y2 = out2_0.x[sort_index], out2_0.y[sort_index]
        d_matrix = torch.abs(out1_0.x-sort_x2)+torch.abs(out1_0.y-sort_y2)
        modify_d_matrix = torch.zeros_like(d_matrix).cuda()
        modify_d_matrix[d_matrix>8.0] = 1.0
        maxi, index = torch.max(modify_d_matrix, 0)
        k_index = torch.gather(sort_index, 0, index.squeeze_().unsqueeze_(0))
        k_index = k_index.squeeze()
        out2_neg = out2_0[k_index].squeeze_()
        zero_loc = torch.nonzero(maxi==0).squeeze()
        out2_neg[zero_loc] = -1*out1_0[zero_loc].clone()
        out2_pos, out1_st = out2_0, out1_0
    else:
        out2_neg = out1_0
        out2_pos, out1_st = out1_0, out1_0
    assert out2_pos.size()==out2_neg.size(), 'positive & nagetive not match'
    assert out1_st.size()==out2_neg.size(), 'target & nagetive not match'
    l1 = torch.where(labels1[batch]>=1.0, torch.tensor(1.0).cuda(), labels1[batch])
    l1 = torch.where(l1==-1.0, torch.tensor(1.0).cuda(), l1)
    l2 = torch.where(labels2[batch]>=1.0, torch.tensor(1.0).cuda(), labels2[batch]) 
    loss_single, loss_feat, loss_det = Loss(l1, l2, out1_st, out1[batch], out2_pos , out2[batch], out2_neg)
    return loss_single, loss_feat, loss_det

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(epoch, model, Loss, train_data,optimizer):
    """训练过程"""
    losses = AverageMeter()
    losses_feat = AverageMeter()
    losses_det = AverageMeter()
    model.train()
    for step, (inputs1, labels1, inputs2, labels2) in enumerate(train_data):
        inputs1 = inputs1.cuda(params['gpu'][0])
        labels1 = labels1.cuda(params['gpu'][0])
        inputs2 = inputs2.cuda(params['gpu'][0])
        labels2 = labels2.cuda(params['gpu'][0])
#        print(torch.from_numpy(np.array(loc1)).size())
#        print(torch.from_numpy(np.array(loc2)).size())
        inputs1.unsqueeze_(1)
        inputs2.unsqueeze_(1)
        #print(torch.nonzero(labels1!=0).size())
        desc1, det1 = model(inputs1)
        desc2, det2 = model(inputs2)
        result = []        
        for batch in range(labels1.size()[0]):
            if batch == 0:
                loss, loss_feat, loss_det = caculate_loss(batch,labels1, labels2, desc1, desc2, det1, det2)      
            else:
                loss.add_(caculate_loss(batch,labels1, labels2, desc1, desc2, det1, det2)[0])
                loss_feat.add_(caculate_loss(batch,labels1, labels2, desc1, desc2, det1, det2)[1])
                loss_det.add_(caculate_loss(batch,labels1, labels2, desc1, desc2, det1, det2)[2])
        loss = loss/batch
        loss_feat = loss_feat/batch
        loss_det = loss_det/batch
        losses.update(loss.item(), inputs1.size(0))
        losses_feat.update(loss_feat.item(), inputs1.size(0))
        losses_det.update(loss_det.item(), inputs1.size(0))        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('-------------------------------------------------------------------------------')
        print('epoch:', epoch, 'step:', step+1,'train loss:%0.4f'%losses.avg, 'train feat loss:%0.4f'%losses_feat.avg, 'train det loss:%0.4f'%losses_det.avg, 'lr:%0.4f'%optimizer.param_groups[0]["lr"])
    return losses.avg, losses_feat.avg, losses_det.avg


def valid(epoch, model, loss, valid_data,optimizer):
    model.eval()
    losses = AverageMeter()
    with torch.no_grad():
        for step, (inputs1, labels1, inputs2, labels2) in enumerate(valid_data):
            inputs1 = inputs1.cuda(params['gpu'][0])
            labels1 = labels1.cuda(params['gpu'][0])
            inputs2 = inputs2.cuda(params['gpu'][0])
            labels2 = labels2.cuda(params['gpu'][0])
            inputs1.unsqueeze_(1)
            inputs2.unsqueeze_(1)
            desc1, det1 = model(inputs1)
            desc2, det2 = model(inputs2)
            for batch in range(labels1.size()[0]):
                if batch == 0:
                    loss, loss_feat, loss_det = caculate_loss(batch,labels1, labels2, desc1, desc2, det1, det2)
                else:
                    loss.add_(caculate_loss(batch,labels1, labels2, desc1, desc2, det1, det2)[0])
                    loss_feat.add_(caculate_loss(batch,labels1, labels2, desc1, desc2, det1, det2)[1])
                    loss_det.add_(caculate_loss(batch,labels1, labels2, desc1, desc2, det1, det2)[2])
            loss = loss/batch
            loss_feat = loss_feat/batch
            loss_det = loss_det/batch
            losses.update(loss.item(), inputs1.size(0))
            print('epoch:', epoch, 'step:', step+1, 'valid loss:', losses.avg)
    return loss, loss_feat, loss_det


if __name__ == '__main__':
    train_data = DataLoader(MYdata(params['imagepath'], params['keypoint'], params['gt'] , mode='train'),batch_size=params['batch_size'], shuffle=True, num_workers=8)
    valid_data = DataLoader(MYdata(params['imagepath'], params['keypoint'], params['gt'], mode='valid'), batch_size=params['batch_size'], shuffle=True, num_workers=8)

    model = GCNnet()
    model.cuda(params['gpu'][0])
    model = nn.DataParallel(model, device_ids=params['gpu'])  # multi-Gpu 
    if params['pretrained']:
         pretrain_dict = torch.load(params['pretrained'], map_location='cpu')
         model_dict = model.state_dict()
         print(pretrain_dict.keys())
         pretrain_dict = {k:v for k,v in pretrain_dict.items() if k in model_dict}
         print(model_dict.keys(),pretrain_dict.keys())
         model_dict.update(pretrain_dict)
         model.load_state_dict(model_dict)
         print('load pretrain model finish')
    train_CE = CeLoss()
    target_CE = CeLoss()
    feat_loss = FeatLoss(1)
    det_loss = DetLoss(train_CE, target_CE)
    Loss = MYLoss(feat_loss, det_loss, params['c'])
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.333, patience=3, verbose=True)
#    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    if os.path.exists(params['log']):
        os.remove(params['log'])
    for n in range(params['num_epoch']):
        loss, loss_feat, loss_det = train(n, model, Loss, train_data,optimizer)
        vloss, vloss_feat, vloss_det = valid(n, model, Loss, valid_data, optimizer)
        scheduler.step(loss)
        torch.save(model.state_dict(),params['model_path'])
        with open(params['log'], 'a') as f:
            f.write('---------------------------------------------------------------------------\r\n')
            f.write('epoch:'+str(n)+'     train loss:%0.4f'%loss+'     valid loss:%0.4f'%vloss+'\r\n')
            f.write('epoch:'+str(n)+'train feat loss:%0.4f'%loss_feat+'valid feat loss:%0.4f'%vloss_feat+'\r\n')
            f.write('epoch:'+str(n)+' train det loss:%0.4f'%loss_det+' valid det loss:%0.4f'%vloss_det+'\r\n')
