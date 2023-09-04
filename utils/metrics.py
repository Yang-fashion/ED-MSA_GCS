from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
from scipy.spatial.distance import directed_hausdorff
import torch
import numpy as np


class MetricTracker(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.avg = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def Hausdorf(output, target):

    temp_preds = output.cpu().detach().numpy().squeeze(0)
    temp_target = target.cpu().detach().numpy().squeeze(0)
    dist1 = directed_hausdorff(temp_preds, temp_target)[0]
    dist2 = directed_hausdorff(temp_target, temp_preds)[0]
    hauss = max(dist1, dist2)

    return hauss


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def dice_coeff(input, target):
    num_in_target = input.size(0)

    smooth = 1.0

    pred = input.view(num_in_target, -1)  # (8, 5760)
    truth = target.view(num_in_target, -1)

    intersection = (pred * truth).sum(1)

    loss = (2.0 * intersection + smooth) / (pred.sum(1) + truth.sum(1) + smooth)

    return loss.mean().item()


# def dice_coeff(output, target):
#     smooth = 1e-5
#
#     output = torch.sigmoid(output).view(-1).data.cpu().numpy()
#     target = target.view(-1).data.cpu().numpy()
#     intersection = (output * target).sum()
#     dice = (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)
#
#     return dice


def BoundF(predict, mask):
    precision = 0
    recall = 0
    accuracy = 0
    batch_size = predict.size()[0]
    for i in range(batch_size):
        if len(torch.unique(mask[i])) == 1:
            continue
        output = predict[i].squeeze(0).reshape([-1]).cpu().detach().numpy()
        label = mask[i].reshape([-1]).cpu().detach().numpy()

        output = output.astype('int64').ravel()
        label = label.astype('int64').ravel()
        pre = precision_score(label, output, average='macro')
        re = recall_score(label, output, average='macro')
        acc = accuracy_score(label, output)

        precision += pre
        recall += re
        accuracy += acc

    precision = precision / batch_size
    recall = recall / batch_size
    accuracy = accuracy / batch_size

    return precision, recall, accuracy


if __name__ == "__main__":
    x = torch.rand(3, 1, 240, 240)
    y = torch.rand(3, 1, 240, 240)
    print(x)
    batch = x.size()[0]
    for i in range(batch):
        # if len(torch.unique(y[i])) == 1:
        #     continue
        x = x[i].squeeze(0).reshape([-1]).astype('int64').ravel()
        y = y[i].reshape([-1]).astype('int64').ravel()
        p, r, a = BoundF(x, y)
        print(p, r, a)
    x = [0, 1, 0, 1]  # 预测的值
    y = [0, 1, 1, 0]  # 真实的值

    p, r, a = BoundF(x, y)
    print(p, r, a)



