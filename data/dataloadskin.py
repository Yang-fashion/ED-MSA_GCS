import numpy as np
from torch.utils.data import Dataset  # 读取数据集
from PIL import Image  # 处理图像
from torchvision import transforms as transforms
from data import data_transforms


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, train, transform=None):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.train = train
        self.transform = transform

    def __getitem__(self, idx):
        img_path = self.imgs_dir[idx]
        mask_path = self.masks_dir[idx]

        image = Image.open(img_path)  #.convert("RGB")
        mask = Image.open(mask_path)

        image = self.preprocess(image)
        mask = self.preprocess(mask)

        if self.transform is not None:
            image, mask = self.train_transform(image, mask)
        if self.train is False:
            image, mask = self.val_transform(image, mask)

        return {'image': image, 'mask': mask}

    def __len__(self):
        return len(self.imgs_dir)

    @classmethod
    def preprocess(cls, pil_img):
        pil_img = pil_img.resize((320, 320))

        img_nd = np.array(pil_img)
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        # img_trans = img_nd.transpose((2, 0, 1))
        # img_trans = img_trans / 255

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
                data_transforms.RandomHorizontalFlip(0.5),
            ])
            sample = data_trans(image, mask)
            image = sample['image']
            mask = sample['mask']
        return image, mask

    def val_transform(self, image, mask):
        image_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        mask_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])
        image = image_transforms(image)
        mask = mask_transforms(mask)
        return image, mask