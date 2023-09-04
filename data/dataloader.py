import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import os.path
from PIL import Image
import cv2
import numpy as np
from data import data_transforms
import mmcv

"""torch.utils.data模块是子类化数据，transforms 库对数据进行预处理"""


# 定义MyTrainData类，继承Dataset方法，并重写__getitem__()和__len__()方法
class MyTrainData(torch.utils.data.Dataset):

    # 初始化函数，得到数据
    def __init__(self, root, train=True, img_scale=0.5, transform=None):
        self.train = train
        self.root = root
        self.images_folder_path = os.path.join(self.root, "imgs")
        self.masks_folder_path = os.path.join(self.root, "masks")

        self.images_path_list = self.read_files_path(self.images_folder_path)
        self.masks_path_list = self.read_files_path(self.masks_folder_path)
        self.scale = img_scale
        self.transform = transform

    # index 是根据batchsize划分数据后得到索引，最后将data和对应的masks一起返回
    def __getitem__(self, index):
        image_path = self.images_path_list[index]
        mask_path = self.masks_path_list[index]

        # 结肠息肉数据集读取方法
        # file_client = mmcv.FileClient(backend='disk')
        # img_bytes = file_client.get(image_path)
        # image = mmcv.imfrombytes(img_bytes, flag='color', backend='tifffile')
        # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path)
        image = self.preprocess(image, self.scale)
        mask = self.preprocess(mask, self.scale)

        if self.transform is not None:
            image, mask = self.train_transform(image, mask)
        if self.train is False:
            image, mask = self.val_transform(image, mask)

        return {'image': image, 'mask': mask}

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.images_path_list)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))
        # pil_img = pil_img.resize((256, 256))

        img_nd = np.array(pil_img)
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        # img_trans = img_nd.transpose((2, 0, 1))
        # img_nd = img_trans / 255

        return img_nd

    def train_transform(self, image, mask):
        image_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        mask_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])
        image = image_transforms(image)
        mask = mask_transforms(mask)
        if self.transform is not None:
            data_trans = data_transforms.Compose([
                data_transforms.CenterCrop(240),    # 结肠息肉数据集中心裁剪大小288
                data_transforms.RandomHorizontalFlip(0.5),
            ])
            sample = data_trans(image, mask)
            image = sample['image']
            mask = sample['mask']
        return image, mask

    def val_transform(self, image, mask):
        image_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(240),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        mask_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(240),
            transforms.ToTensor()
        ])
        image = image_transforms(image)
        mask = mask_transforms(mask)
        return image, mask

    def read_files_path(self, folder_path):  # 截图1
        files_names_list = os.listdir(folder_path)  # os.listdir()返回folder_path所指向的文件夹下所有文件的名称
        files_paths_list = [os.path.join(folder_path, file_name) for file_name in files_names_list]  # 路径拼接
        return files_paths_list


if __name__ == '__main__':

    file_client = mmcv.FileClient(backend='disk')
    img_bytes = file_client.get('/root/projects/Datasets/polyp_colon/test/imgs/12.tif')
    image = mmcv.imfrombytes(img_bytes, flag='color', backend='tifffile')

    print('debugger')
