import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import argparse
import warnings

from backbones.network import MyUnet
from backbones.unet import UNet
from backbones.attunet import AttUNet
from backbones.Nestedunet import NestedUNet
from backbones.scseUnet import scseUNet
from backbones.CANet.CANet import CAUnet
from backbones.UNext import UNext
from data.dataloader import MyTrainData
from utils import myloss
from utils import loss_function
from utils import caLoss
from utils import metrics
from show.saver import Saver
from show.summaries import TensorboardSummary
from thop import profile


warnings.filterwarnings('ignore')
device = 'cuda'if torch.cuda.is_available() else 'cpu'


def get_args():
    parser = argparse.ArgumentParser(description='Train a segmentation')
    # Training
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--start-epoch', type=int, default=0, help='Starting epoch')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr-step', type=int, default=200, help='LR scheduler step')

    # Architecture
    parser.add_argument('--arch', type=str, default='mynet',
                        choices=['myunet', 'unet', 'attunet', 'unet++', 'scseUnet', 'unext'],
                        help='Network architecture. unet or resnet')
    parser.add_argument('--ablation', type=str, default='bd_loss', choices=['activeLoss', 'bd_loss', 'BCEDLoss'],
                        help='Network architecture.')

    # Data
    parser.add_argument('--train-dataset', type=str, default='COVID-19', help='Training dataset')
    parser.add_argument('--data-path', type=str, default='/root/projects/Datasets/COVID-19',
                        help='Path to buildings dataset directory')
    parser.add_argument('--scale', '-s', default=0.5, type=float, help="Scale factor for the input images")

    # Misc
    parser.add_argument('--eval-rate', type=int, default=1, help='Evaluate after eval_rate epochs')
    parser.add_argument('--save-rate', type=int, default=1, help='Save rate is save_rate * eval_rate')
    parser.add_argument('--resume', type=str, default=None, help='Resume file path')
    args = parser.parse_args()

    return args


def train_epoch(model, criterion, optimizer, data_loader, epoch, device, summary):
    model.train()
    iterator = tqdm(data_loader)

    for idx, image_and_mask in enumerate(iterator):
        image = image_and_mask['image'].to(device, dtype=torch.float32)
        mask = image_and_mask['mask'].to(device, dtype=torch.float32)

        mask_pred = model(image)
        # flops, params = profile(model, image)
        # print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

        # 获取loss
        loss = criterion(mask_pred, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrices
        train_dice = metrics.dice_coeff(mask_pred, mask)
        train_iou = metrics.iou_score(mask_pred, mask)
        precision, recall, accuracy = metrics.BoundF(mask_pred, mask)

        iterator.set_description(
            '(train | {}) Epoch [{epoch}/{epochs}] :: Loss={loss:.4f} | Dice={dice:.4f} | precision={precision:.4f}'.format(
                args.arch, epoch=epoch + 1, epochs=args.epochs, loss=loss.item(), dice=train_dice, precision=precision,
            ))

        global_step = epoch * len(data_loader) + idx
        summary.add_scalar('Train/loss', loss.item(), global_step)
        summary.add_scalar('Train/dice', train_dice, global_step)
        summary.add_scalar('Train/iou', train_iou, global_step)
        summary.visualize_image('Train', image, mask, mask_pred, mask_pred, global_step)


def val_epoch(model, data_loader, epoch, device, summary):
    model.eval()
    iterator = tqdm(data_loader)

    valuate_dice = metrics.MetricTracker()
    valuate_huass = metrics.MetricTracker()
    valuate_iou = metrics.MetricTracker()
    precision, recall, accuracy = 0, 0, 0

    for idx, image_and_mask in enumerate(data_loader):
        image = image_and_mask['image'].to(device, dtype=torch.float32)
        mask_type = torch.float32 if model.out_channels == 1 else torch.long
        mask_gt = image_and_mask['mask'].to(device, dtype=mask_type)

        mask_pred = model(image)

        acm_pred = mask_pred
        mask = mask_gt

        dice = metrics.dice_coeff(acm_pred, mask)
        valuate_dice.update(dice)
        valuate_iou.update(metrics.iou_score(acm_pred, mask))
        precision, recall, accuracy = metrics.BoundF(acm_pred, mask)
        hausdorff = 0

        for mask_pred, mask_gt in zip(mask_pred, mask_gt):
            mask_pred = (mask_pred > 0.5).float()  # mask_pred.size(1,480,480)
            haus = metrics.Hausdorf(mask_pred, mask_gt)
            hausdorff = haus
            valuate_huass.update(metrics.Hausdorf(mask_pred, mask_gt), mask_pred.size(0))
            # valuate_f1.update(metrics.hunxiao(mask_pred, mask_gt), mask_pred.size(0))

        iterator.set_description(
            '(val | {}) Epoch [{epoch}/{epochs}] :: Dice={dice:.4f} | Haus={hauss:.4f}'.format(
                args.arch, epoch=epoch + 1, epochs=args.epochs, dice=dice, hauss=hausdorff
            ))
        global_step = epoch * len(data_loader) + idx
        summary.add_scalar('Validation/dice_iter', dice, global_step)
        summary.visualize_image('Val', image, mask, acm_pred, acm_pred, global_step)

    valuate_dice = valuate_dice.avg
    valuate_huass = valuate_huass.avg
    valuate_iou = valuate_iou.avg

    return valuate_dice, valuate_huass, valuate_iou, precision, recall, accuracy


if __name__=='__main__':
    args = get_args()

    torch.multiprocessing.set_start_method('spawn')

    # Define Saver
    saver = Saver(args)
    saver.save_experiment_config()
    # Define Tensorboard Summary
    summary = TensorboardSummary(saver.experiment_dir)
    args.exp = saver.experiment_dir.split('_')[-1]

    train_data = MyTrainData(args.data_path+'/train', train=True, img_scale=args.scale, transform=True)
    val_data = MyTrainData(args.data_path+'/test', train=False, img_scale=args.scale)

    train_dl = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = 'cuda'if torch.cuda.is_available() else 'cpu'

    model = MyUnet(3, 1).to(device)
    criterion = loss_function.BCEDiceLoss()

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=1e-8)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=256, gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step, gamma=0.1)

    if args.resume:
        if args.resume == 'best':
            args.resume = os.path.join(saver.directory, 'model_best.pth')
        if not os.path.isfile(args.resume):
            raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        best_pred = checkpoint['best_pred']

    best_dice = 0
    log = pd.DataFrame(index=[],
                       columns=['epoch', 'val_dice', 'val_hausdorff', 'val_iou', 'precision', 'recall', 'accuracy'])
    for epoch in range(args.start_epoch, args.epochs):
        print("----> starting train")
        train_epoch(model, criterion, optimizer, train_dl, epoch, device, summary)
        # scheduler.step()

        dice, haus, iou, precision, recall, accuracy = val_epoch(model, val_dl, epoch, device, summary)
        summary.add_scalar('Validation/dice', dice, epoch)
        summary.add_scalar('Validation/Hausdorff', haus, epoch)
        summary.add_scalar('Validation/iou', iou, epoch)

        tmp = pd.Series([epoch + 1, dice, haus, iou, precision, recall, accuracy],
                        index=['epoch', 'val_dice', 'val_hausdorff', 'val_iou', 'precision', 'recall', 'accuracy'])
        log = log.append(tmp, ignore_index=True)
        log.to_csv(saver.experiment_dir + '/log.csv', index=False)

        is_best = False
        if dice > best_dice:
            best_dice = dice
            is_best = True
            print('==> save best model  --best is Dice:{:.4f}'.format(best_dice))

            model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_pred': best_dice,
            }, is_best)

