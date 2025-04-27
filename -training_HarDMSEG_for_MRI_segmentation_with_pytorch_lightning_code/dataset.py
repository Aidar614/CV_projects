from torch.utils.data import DataLoader
import pytorch_lightning as pl
import cv2
from torch.utils.data import Dataset
import os
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import config


class LungSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        
        image_path = os.path.join(self.image_dir, image_name)

        parts = image_name.split('_')
        mask_index = int(parts[-1].split('.')[0])  
        
        base_name = '_'.join(parts[:-1])  
        mask_name = f"{base_name}_mask_{mask_index}.jpg"
        mask_path = os.path.join(self.mask_dir, mask_name)

        img = cv2.imread(image_path)
        mask = cv2.imread(mask_path)

        img = np.array(img, dtype=np.float32)

        mask_3ch = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        
        mask_condition = np.all(mask < 50, axis=-1)

        mask_3ch[mask_condition, 0] = 1

        max_color_index = np.argmax(mask, axis=-1)
        rows, cols = np.indices(mask.shape[:2]) 
        mask_3ch[rows, cols, max_color_index + 1] = 1

        
        if self.transform is not None:
            augmentations = self.transform(image=img, mask=mask_3ch)
            image = augmentations["image"]
            mask = augmentations["mask"]
        
        mask = mask.permute(2, 0, 1) 
        return image, mask


class DataModule(pl.LightningDataModule):
    def __init__(self, train_dir, train_maskdir, val_dir, val_maskdir,  batch_size, num_workers):
        super().__init__()
        self. train_dir =  train_dir
        self.train_maskdir= train_maskdir
        self.val_dir = val_dir
        self.val_maskdir = val_maskdir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = A.Compose(
        [
            A.Resize(height=config.IMAGE_HEIGHT, width=config.IMAGE_WIDTH),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.1),
            A.RandomBrightnessContrast(p=0.3),  
            A.GaussianBlur(p=0.2),  
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
            ToTensorV2(),
        ],
    )

        self.val_transforms = A.Compose(
            [   
                A.Resize(height=config.IMAGE_HEIGHT, width=config.IMAGE_WIDTH),
                ToTensorV2(),
            ],
        )

    def setup(self, stage):
        self.train_ds = LungSegmentationDataset(
        image_dir=self.train_dir,
        mask_dir=self.train_maskdir,
        transform=self.train_transform,
    )

        self.val_ds = LungSegmentationDataset(
            image_dir=self.val_dir,
            mask_dir=self.val_maskdir,
            transform=self.val_transforms,
        )

        
    def train_dataloader(self):
        return DataLoader(
            self.train_ds_ds,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            shuffle=False,
        )
