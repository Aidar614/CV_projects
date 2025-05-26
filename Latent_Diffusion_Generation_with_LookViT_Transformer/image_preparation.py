import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from diffusers import AutoencoderKL
import numpy as np
from tqdm import tqdm

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        from PIL import Image
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        return self.transform(image), self.image_files[idx]
if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    image_dir = 'datasets/images'
    output_dir = 'latent_output'
    os.makedirs(output_dir, exist_ok=True)

    dataset = ImageDataset(image_dir, transform)
    dataloader = DataLoader(dataset, batch_size=6, num_workers=12, shuffle=False)

    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", use_safetensors=True).to('cuda')

    for batch, filenames in tqdm(dataloader, desc="Processing images"):
        batch = batch.to('cuda')
        with torch.no_grad():
            latent_features = vae.encode(batch).latent_dist.sample().mul_(vae.config.scaling_factor).cpu().numpy()
        for latent, fname in zip(latent_features, filenames):
            np.save(os.path.join(output_dir, f"{os.path.splitext(fname)[0]}.npy"), latent)
    print(latent_features.shape)