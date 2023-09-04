import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = inputs[:, 2:, :, :]
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class log_cosh_diceloss(nn.Module):
    def __init__(self):
        super(log_cosh_diceloss, self).__init__()

    def forward(self, inputs, target, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = target.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        x = 1 - dice
        x_log = torch.log((torch.exp(x) + torch.exp(-x)) / 2.0)
        return x_log


# class BCEDiceLoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super().__init__()
#
#     def forward(self, input, target):
#         pred = input.view(-1)
#         truth = target.view(-1)
#
#         # BCE loss
#         bce_loss = nn.BCELoss()(pred, truth).double()
#
#         # Dice Loss
#         dice_coef = (2.0 * (pred * truth).double().sum() + 1) / (
#             pred.double().sum() + truth.double().sum() + 1
#         )
#
#         return bce_loss + (1 - dice_coef)


class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, alpha_t=0.5, beta_t=0.5):
        super(TverskyLoss, self).__init__()
        self.alpha_t = alpha_t
        self.beta_t = beta_t

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + self.alpha_t * FP + self.beta_t * FN + smooth)
        loss = 1 - Tversky
        # print(self.alpha_t, self.beta_t)

        return loss


class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, alpha_f=0.8, gamma_f=2.0):
        super(FocalLoss, self).__init__()
        self.alpha_f = alpha_f
        self.gamma_f = gamma_f

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # first compute binary cross-entropy
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.alpha_f * (1 - BCE_EXP) ** self.gamma_f * BCE

        return focal_loss


class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, alpha_ft=0.5, beta_ft=0.5, gamma_ft=1.0):
        super(FocalTverskyLoss, self).__init__()
        self.alpha_ft = 0.5
        self.beta_ft = 0.5
        self.gamma_ft = 1.0

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + self.alpha_ft * FP + self.beta_ft * FN + smooth)
        FocalTversky = (1 - Tversky) ** self.gamma_ft

        return FocalTversky

