import os
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter


class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory
        self.writer = SummaryWriter(log_dir=os.path.join(self.directory))

    def add_scalar(self, *args):
        self.writer.add_scalar(*args)

    def visualize_image(self, label, image_tgt, mask_gt, cnn_pred, acm_pred, global_step):
        grid_image = make_grid(image_tgt[:3, :, :, :].clone().cpu().data, 3, normalize=True)
        self.writer.add_image('{}/Image'.format(label), grid_image, global_step)

        grid_image = make_grid(mask_gt[:3, :, :, :].clone().cpu().data, 3, normalize=True)
        self.writer.add_image('{}/Mask GT'.format(label), grid_image, global_step)

        grid_image = make_grid(cnn_pred[:3, :, :, :].clone().cpu().data, 3, normalize=True)
        self.writer.add_image('{}/Mask CNN'.format(label), grid_image, global_step)

        grid_image = make_grid(acm_pred[:3, :, :, :].clone().cpu().data, 3, normalize=True)
        self.writer.add_image('{}/Mask ACM'.format(label), grid_image, global_step)

