import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import distance_transform_edt as distance


def softmax_helper(x):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/nd_softmax.py
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)


def one_hot2dist(seg: np.ndarray):
    C: int = len(seg)

    res = np.zeros_like(seg)
    for c in range(C):
        posmask = seg[c].astype(np.bool)

        if posmask.any():
            negmask = ~posmask
            res[c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res


class MyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, alpha_ft=0.5, beta_ft=0.5, gamma_ft=1.0):
        super(MyLoss, self).__init__()
        self.alpha_ft = 0.5
        self.beta_ft = 0.5
        self.gamma_ft = 1.0

    def forward(self, output, target):
        net_output = softmax_helper(output)
        gt_temp = target.cpu().numpy()
        with torch.no_grad():
            dist = one_hot2dist(gt_temp)
        dist = torch.from_numpy(dist).cuda(0)

        pc = net_output[:, 0:, ...].type(torch.float32)
        dc = dist[:, 0:, ...].type(torch.float32)

        multipled = torch.einsum("bcxy,bcxy->bcxy", pc, dc)
        bd_loss = 0.01 * multipled.mean()

        # comment out if your model contains a sigmoid or equivalent activation layer
        out = torch.sigmoid(output)
        # flatten label and prediction tensors
        outputs = out.view(-1)
        targets = target.view(-1)

        # BCEDiceLoss
        bce_loss = nn.BCELoss()(outputs, targets).double()
        dice_coef = (2.0 * (outputs * targets).double().sum() + 1) / (
                outputs.double().sum() + targets.double().sum() + 1)
        BCEDiceLoss = bce_loss + (1 - dice_coef)

        myloss = 0.5*BCEDiceLoss + 0.01*bd_loss

        return bd_loss


class ActiveLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, alpha_ft=0.5, beta_ft=0.5, gamma_ft=1.0):
        super(ActiveLoss, self).__init__()
        self.alpha_ft = 0.5
        self.beta_ft = 0.5
        self.gamma_ft = 1.0

    def forward(self, output, target):
        pred = output.view(-1)
        truth = target.view(-1)

        # length term
        delta_r = output[:, :, 1:, :] - output[:, :, :-1, :]  # horizontal gradient (B, C, H-1, W)
        delta_c = output[:, :, :, 1:] - output[:, :, :, :-1]  # vertical gradient   (B, C, H,   W-1)
        delta_r = delta_r[:, :, 1:, :-2] ** 2  # (B, C, H-2, W-2)
        delta_c = delta_c[:, :, :-2, 1:] ** 2  # (B, C, H-2, W-2)
        delta_pred = torch.abs(delta_r + delta_c)

        epsilon = 1e-8  # where is a parameter to avoid square root is zero in practice.
        x = delta_pred + epsilon
        lenth = torch.mean(torch.sqrt(x))  # eq.(11) in the paper, mean is used instead of sum.

        # region term
        c_in = torch.ones_like(output)
        c_out = torch.zeros_like(output)
        region_in = torch.mean(output * (target - c_in) ** 2)  # equ.(12) in the paper, mean is used instead of sum.
        region_out = torch.mean((1 - output) * (target - c_out) ** 2)
        region = region_in + region_out
        active_loss = 10 * lenth + region

        return active_loss
