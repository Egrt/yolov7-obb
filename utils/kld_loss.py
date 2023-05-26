'''
Author: [egrt]
Date: 2023-01-30 18:47:24
LastEditors: Egrt
LastEditTime: 2023-05-26 15:00:14
Description: 
'''
import torch
import torch.nn as nn

class KLDloss(nn.Module):

    def __init__(self, taf=1.0, fun="sqrt"):
        super(KLDloss, self).__init__()
        self.fun = fun
        self.taf = taf
        self.eps = 1e-8

    def forward(self, pred, target): # pred [[x,y,w,h,angle], ...]
        #assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 5)
        target = target.view(-1, 5)

        delta_x = pred[:, 0] - target[:, 0]
        delta_y = pred[:, 1] - target[:, 1]
        pre_angle_radian = pred[:, 4]
        targrt_angle_radian = target[:, 4]
        delta_angle_radian = pre_angle_radian - targrt_angle_radian

        kld =  0.5 * (
                        4 * torch.pow( ( delta_x.mul(torch.cos(targrt_angle_radian)) + delta_y.mul(torch.sin(targrt_angle_radian)) ), 2) / torch.pow(target[:, 2], 2)
                      + 4 * torch.pow( ( delta_y.mul(torch.cos(targrt_angle_radian)) - delta_x.mul(torch.sin(targrt_angle_radian)) ), 2) / torch.pow(target[:, 3], 2)
                     )\
             + 0.5 * (
                        torch.pow(pred[:, 3], 2) / torch.pow(target[:, 2], 2) * torch.pow(torch.sin(delta_angle_radian), 2)
                      + torch.pow(pred[:, 2], 2) / torch.pow(target[:, 3], 2) * torch.pow(torch.sin(delta_angle_radian), 2)
                      + torch.pow(pred[:, 3], 2) / torch.pow(target[:, 3], 2) * torch.pow(torch.cos(delta_angle_radian), 2)
                      + torch.pow(pred[:, 2], 2) / torch.pow(target[:, 2], 2) * torch.pow(torch.cos(delta_angle_radian), 2)
                     )\
             + 0.5 * (
                        torch.log(torch.pow(target[:, 3], 2) / torch.pow(pred[:, 3], 2))
                      + torch.log(torch.pow(target[:, 2], 2) / torch.pow(pred[:, 2], 2))
                     )\
             - 1.0

        

        if self.fun == "sqrt":
            kld = kld.clamp(1e-7).sqrt()
        elif self.fun == "log1p":
            kld = torch.log1p(kld.clamp(1e-7))
        else:
            pass

        kld_loss = 1 - 1 / (self.taf + self.eps + kld)

        return kld_loss
    
def compute_kld_loss(targets, preds,taf=1.0,fun='sqrt'):
    with torch.no_grad():
        kld_loss_ts_ps = torch.zeros(0, preds.shape[0], device=targets.device)
        for target in targets:
            target = target.unsqueeze(0).repeat(preds.shape[0], 1)
            kld_loss_t_p = kld_loss(preds, target,taf=taf, fun=fun)
            kld_loss_ts_ps = torch.cat((kld_loss_ts_ps, kld_loss_t_p.unsqueeze(0)), dim=0)
    return kld_loss_ts_ps


def kld_loss(pred, target, taf=1.0, fun='sqrt'):  # pred [[x,y,w,h,angle], ...]
    #assert pred.shape[0] == target.shape[0]

    pred = pred.view(-1, 5)
    target = target.view(-1, 5)

    delta_x = pred[:, 0] - target[:, 0]
    delta_y = pred[:, 1] - target[:, 1]
    pre_angle_radian = pred[:, 4]  #3.141592653589793 * pred[:, 4] / 180.0
    targrt_angle_radian = target[:, 4] #3.141592653589793 * target[:, 4] / 180.0
    delta_angle_radian = pre_angle_radian - targrt_angle_radian

    kld = 0.5 * (
            4 * torch.pow((delta_x.mul(torch.cos(targrt_angle_radian)) + delta_y.mul(torch.sin(targrt_angle_radian))),
                          2) / torch.pow(target[:, 2], 2)
            + 4 * torch.pow((delta_y.mul(torch.cos(targrt_angle_radian)) - delta_x.mul(torch.sin(targrt_angle_radian))),
                            2) / torch.pow(target[:, 3], 2)
    ) \
          + 0.5 * (
                  torch.pow(pred[:, 3], 2) / torch.pow(target[:, 2], 2) * torch.pow(torch.sin(delta_angle_radian), 2)
                  + torch.pow(pred[:, 2], 2) / torch.pow(target[:, 3], 2) * torch.pow(torch.sin(delta_angle_radian), 2)
                  + torch.pow(pred[:, 3], 2) / torch.pow(target[:, 3], 2) * torch.pow(torch.cos(delta_angle_radian), 2)
                  + torch.pow(pred[:, 2], 2) / torch.pow(target[:, 2], 2) * torch.pow(torch.cos(delta_angle_radian), 2)
          ) \
          + 0.5 * (
                  torch.log(torch.pow(target[:, 3], 2) / torch.pow(pred[:, 3], 2))
                  + torch.log(torch.pow(target[:, 2], 2) / torch.pow(pred[:, 2], 2))
          ) \
          - 1.0

    if fun == "sqrt":
        kld = kld.clamp(1e-7).sqrt()
    elif fun == "log1p":
        kld = torch.log1p(kld.clamp(1e-7))
    else:
        pass

    kld_loss = 1 - 1 / (taf + kld)
    return kld_loss

if __name__ == '__main__':
    '''
        测试损失函数
    '''
    kld_loss_n = KLDloss(alpha=1,fun='log1p')
    pred = torch.tensor([[5, 5, 5, 23, 0.15],[6,6,5,28,0]]).type(torch.float32)
    target = torch.tensor([[5, 5, 5, 24, 0],[6,6,5,28,0]]).type(torch.float32)
    kld = kld_loss_n(target,pred)