import argparse
import torch
from torchvision import transforms
import os
import time
import numpy as np
from tqdm import tqdm
from PIL import Image
from glob import glob

from backbones.network import MyUnet
from backbones.UNext import UNext
from utils.DCAN_CVV import ACM
from utils import metrics
from data import dataloader
import config
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_args():
    parser = argparse.ArgumentParser(description='Test model')
    parser.add_argument('--mask-threshold', type=float, default=0.5,
                        help="Minimum probability value to consider a mask pixel white")
    parser.add_argument('--scale', '-s', default=config.scale, type=float, help="Scale factor for the input images")
    return parser.parse_args()


def test_img(model, img, scale, out_threshold=0.5):
    model.eval()

    image = torch.from_numpy(dataloader.MyTrainData.preprocess(img, scale))
    # print(image.shape)
    # image = image.unsqueeze(0).to(device, dtype=torch.float32)
    # print(image.shape)

    with torch.no_grad():
        trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        image = trans(image)
        # pred = ACM(pred, image, mask_gt, 3)
        image = image.unsqueeze(0).to(device, dtype=torch.float32)
        output = model(image)
        probs = output.squeeze(0)

        # probs = trans(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()
        res = np.int64(full_mask > out_threshold)

    return res


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


def mask_resize(mask):
    w, h = mask.size
    newW, newH = int(args.scale * w), int(args.scale * h)
    assert newW > 0 and newH > 0, 'Scale is too small'
    pil_img = mask.resize((newW, newH))
    mask_nd = np.array(pil_img)
    mask_nd = mask_nd.astype('float32') / 255
    # mask_trans = mask_nd.transpose((2, 0, 1))
    # # mask_s = mask_trans.astype('float32') / 255
    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    mask_re = trans(mask_nd)
    mask_re = mask_re.cpu().numpy()

    return mask_re


if __name__ == "__main__":
    result_img_path = "outputs/cell/"
    checkpoint = '/root/projects/myself/runs/cell/unext/exp_2022-06-24_10:32 0.8744/checkpoint.pth'
    if not os.path.exists(result_img_path):
        os.makedirs(result_img_path)

    args = get_args()
    start = time.perf_counter()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = UNext(3, 1).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device)['state_dict'])  # ['state_dict']
    print("Model loaded !")

    test_img_path = glob(config.test_img_path + "/*")
    test_mask_path = glob(config.test_mask_path + "/*")

    iou = metrics.MetricTracker()
    dice = metrics.MetricTracker()
    # dice = []
    acc = []
    recall = []
    spec = []
    pres = []
    f1s = []

    for i in tqdm(range(len(test_img_path))):
        image = Image.open(test_img_path[i]).convert('RGB')
        mask_gt = Image.open(test_mask_path[i])

        mask = mask_resize(mask_gt)

        pred = test_img(model, image, scale=args.scale, out_threshold=args.mask_threshold)

        result = mask_to_image(pred)
        result.save(result_img_path + os.path.basename(test_img_path[i]))

        iou.update(metrics.iou_score(pred, mask))
        # dice.update(metrics.dice_coeff(pred, mask))

        # accuracy, sensitivity, specificity, precision, f1 = metrics.hunxiao(pred, mask)
        # acc.append(accuracy)
        # recall.append(sensitivity)
        # spec.append(specificity)
        # pres.append(precision)
        # f1s.append(f1)

    print("Iou: {:.4f}".format(iou.avg))
    print("Dice: {:.4f}".format(dice.avg))
    print('Accuracy:{:.4f}' .format(np.mean(acc)))
    print('Recall:{:.4f}' .format(np.mean(recall)))
    print('Specificity:{:.4f}' .format(np.mean(spec)))
    print('Precision:{:.4f}' .format(np.mean(pres)))
    print('f1:{:.4f}' .format(np.mean(f1s)))
    end = time.perf_counter()
    print("Running time: %s Seconds " % (end - start))

