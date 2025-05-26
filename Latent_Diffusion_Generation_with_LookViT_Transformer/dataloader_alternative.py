import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image 
from torchvision import transforms

class LatentDataset(Dataset):
    def __init__(self, latent_dir, image_dir): 
        self.latent_dir = latent_dir
        self.image_dir = image_dir
        self.latent_files = sorted(os.listdir(latent_dir))
      
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        assert len(self.latent_files) == len(self.image_files), "Количество латентов и изображений должно совпадать"
        for latent_file, image_file in zip(self.latent_files, self.image_files):
            assert os.path.splitext(latent_file)[0] == os.path.splitext(image_file)[0], \
                f"Имена файлов не совпадают: {latent_file} и {image_file}"

    def __len__(self):
        return len(self.latent_files)

    def __getitem__(self, idx):
        latent_file = self.latent_files[idx]
        image_file = self.image_files[idx]
      
        latent = np.load(os.path.join(self.latent_dir, latent_file))
       
        image = Image.open(os.path.join(self.image_dir, image_file)).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((96, 96)), 
            transforms.ToTensor(), 
        ])
        image = transform(image) 
        image = image * 2 - 1 
        return torch.tensor(latent, dtype=torch.float32), image